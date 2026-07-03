"""
run.py — A6 Speech Processing submission entry point.

Usage examples (see README.md for full command list and results):

    python3 run.py --model ctc --epochs 300 --train
    python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train
    python3 run.py --model voice-clone --extract-se --reference my_voice.wav
    python3 run.py --model voice-clone --accent us --text "I got the job!" --generate
    python3 run.py --model voice-clone --accent all --text "Hello world" --generate
    python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── ffmpeg/ffprobe PATH fix ────────────────────────────────────────
# On shared servers without sudo, system ffmpeg/ffprobe may be missing.
# `pip install static-ffmpeg` provides static binaries; this makes sure
# they're discoverable on PATH before any audio libraries are imported.
try:
    import static_ffmpeg
    _ffmpeg_path, _ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    _bin_dir = os.path.dirname(_ffmpeg_path)
    if _bin_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _bin_dir + os.pathsep + os.environ.get('PATH', '')
except ImportError:
    pass  # static-ffmpeg not installed; assume system ffmpeg/ffprobe are already on PATH
# ────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('data/speech', exist_ok=True)
os.makedirs('data/voice_clone', exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# Part 3 / Exercise 2 — Toy CTC model
# ────────────────────────────────────────────────────────────────────

ALPHABET = list('helo wrd')
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1
N_MELS = 20
WORDS = ['hello', 'world', 'hero', 'red', 'led', 'doer']
BLANK = '_'


def ctc_collapse(alignment):
    merged = []
    for ch in alignment:
        if not merged or ch != merged[-1]:
            merged.append(ch)
    return ''.join(ch for ch in merged if ch != BLANK)


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def cer(pred, true):
    if len(true) == 0:
        return 0.0
    return levenshtein(pred, true) / len(true)


def synthesize_frames(word, frames_per_char=(3, 8)):
    frames = []
    for ch in word:
        n = random.randint(*frames_per_char)
        base = np.zeros(N_MELS)
        base[CHAR2IDX[ch] % N_MELS] = 3.0
        for _ in range(n):
            frames.append(base + np.random.randn(N_MELS) * 0.5)
    return np.stack(frames)


class TinyCTCModel(nn.Module):
    def __init__(self, in_dim=N_MELS, hidden=64, vocab=VOCAB_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, vocab)

    def forward(self, x):
        h, _ = self.lstm(x)
        return F.log_softmax(self.fc(h), dim=-1)


def train_ctc(epochs=300, frames_per_char=(3, 8), lr=1e-2):
    model = TinyCTCModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    loss_history, cer_history = [], []
    for step in range(epochs):
        word = random.choice(WORDS)
        frames = synthesize_frames(word, frames_per_char)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long)

        log_probs = model(x).transpose(0, 1)
        input_lengths = torch.tensor([log_probs.size(0)])
        target_lengths = torch.tensor([len(targets)])

        loss = ctc_loss_fn(log_probs.cpu(), targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        with torch.no_grad():
            pred_ids = log_probs.squeeze(1).argmax(dim=-1).cpu().tolist()
            pred_chars = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
            decoded = ctc_collapse(pred_chars)
        cer_history.append(cer(decoded, word))

        if (step + 1) % 50 == 0:
            print(f'Step {step + 1:3d} | CTC loss: {np.mean(loss_history[-50:]):.4f} '
                  f'| CER: {np.mean(cer_history[-50:]):.4f}')

    print(f'\nFinal 50-step avg CER: {np.mean(cer_history[-50:]):.4f}')
    return model, loss_history, cer_history


# ────────────────────────────────────────────────────────────────────
# Part 4 / Exercise 3 — wav2vec2 linear probe vs. raw-mel baseline
# ────────────────────────────────────────────────────────────────────

def run_wav2vec2_probe(classes, n_per_class=40):
    import torchaudio
    import torchaudio.transforms as TA
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from sklearn.model_selection import train_test_split

    probe_words = classes
    w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    w2v_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(DEVICE).eval()
    for p in w2v_model.parameters():
        p.requires_grad = False

    os.makedirs('data/speechcommands', exist_ok=True)
    sc_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='data/speechcommands', download=True)

    by_label = {w: [] for w in probe_words}
    for i in range(len(sc_dataset)):
        wvf, sr, label, *_ = sc_dataset[i]
        if label in by_label and len(by_label[label]) < n_per_class:
            by_label[label].append(wvf)
        if all(len(v) >= n_per_class for v in by_label.values()):
            break

    def extract_w2v_feats():
        feats, labels_list = [], []
        with torch.no_grad():
            for label, clips in by_label.items():
                for wvf in clips:
                    inputs = w2v_extractor(wvf.squeeze(0).numpy(), sampling_rate=16000,
                                            return_tensors='pt').to(DEVICE)
                    out = w2v_model(**inputs).last_hidden_state
                    feats.append(out.mean(dim=1).squeeze(0).cpu())
                    labels_list.append(probe_words.index(label))
        return torch.stack(feats), torch.tensor(labels_list)

    def extract_mel_feats():
        mel_tf = TA.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80)
        feats = []
        for label, clips in by_label.items():
            for wvf in clips:
                mel = mel_tf(wvf)
                log_mel = torch.log(mel + 1e-9)
                feats.append(log_mel.mean(dim=-1).squeeze(0))
        return torch.stack(feats)

    def linear_probe_acc(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=0.3, random_state=42, stratify=y.numpy())
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        probe = nn.Linear(X.shape[1], len(probe_words))
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(100):
            logits = probe(X_train_t)
            loss = F.cross_entropy(logits, y_train_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (probe(X_test_t).argmax(1) == y_test_t).float().mean().item()
        return acc

    X_w2v, y = extract_w2v_feats()
    X_mel = extract_mel_feats()

    acc_w2v = linear_probe_acc(X_w2v, y)
    acc_mel = linear_probe_acc(X_mel, y)

    print(f'Classes: {probe_words}')
    print(f'Raw mel-spectrogram linear probe accuracy: {acc_mel * 100:.1f}%')
    print(f'wav2vec2 (frozen) linear probe accuracy:    {acc_w2v * 100:.1f}%')
    print(f'Random baseline: {100 / len(probe_words):.1f}%')
    return acc_mel, acc_w2v


# ────────────────────────────────────────────────────────────────────
# Part 5 / Exercise 4 — Voice cloning with OpenVoice
# ────────────────────────────────────────────────────────────────────

def _load_openvoice():
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter

    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2')
    converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device=str(DEVICE))
    converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    return converter, ckpt_dir


def extract_se(reference_path):
    from openvoice import se_extractor
    converter, ckpt_dir = _load_openvoice()
    target_se, _ = se_extractor.get_se(
        reference_path, converter, target_dir='data/voice_clone/processed', vad=True)
    torch.save(target_se, 'data/voice_clone/target_se.pth')
    print(f'Extracted tone color embedding: shape {target_se.shape}')
    print('Saved to data/voice_clone/target_se.pth')
    return target_se


STYLE_TO_SE = {
    'us': ('en-us.pth', 'EN-US'),
    'br': ('en-br.pth', 'EN-BR'),
    'india': ('en-india.pth', 'EN_INDIA'),
    'au': ('en-au.pth', 'EN-AU'),
}

CROSS_LINGUAL_SE = {
    'en': 'en-default.pth',
    'es': 'es.pth',
    'fr': 'fr.pth',
    'zh': 'zh.pth',
    'jp': 'jp.pth',
    'kr': 'kr.pth',
}


def generate_voice(text, accent=None, language=None):
    from melo.api import TTS

    converter, ckpt_dir = _load_openvoice()
    if not os.path.exists('data/voice_clone/target_se.pth'):
        raise FileNotFoundError(
            'No saved tone color embedding found. Run --extract-se first.')
    target_se = torch.load('data/voice_clone/target_se.pth', map_location=DEVICE)

    if language is not None:
        lang_code = language.upper()
        base_tts = TTS(language=lang_code, device=str(DEVICE))
        spk_ids = base_tts.hps.data.spk2id
        spk_key = list(spk_ids.keys())[0]

        base_path = f'data/voice_clone/base_{language}.wav'
        out_path = f'data/voice_clone/cloned_{language}.wav'
        base_tts.tts_to_file(text, spk_ids[spk_key], base_path, speed=1.0)

        se_file = CROSS_LINGUAL_SE.get(language.lower(), f'{language.lower()}.pth')
        source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/{se_file}', map_location=DEVICE)
        converter.convert(audio_src_path=base_path, src_se=source_se,
                           tgt_se=target_se, output_path=out_path)
        print(f'[{language}] "{text}" -> {out_path}')
        return [out_path]

    base_speaker_tts = TTS(language='EN', device=str(DEVICE))
    speaker_ids = base_speaker_tts.hps.data.spk2id

    accents = list(STYLE_TO_SE.keys()) if accent == 'all' else [accent]
    out_paths = []
    for acc in accents:
        se_file, spk_key = STYLE_TO_SE[acc]
        spk_id = speaker_ids[spk_key] if spk_key in speaker_ids else speaker_ids['EN-US']

        base_path = f'data/voice_clone/base_{acc}.wav'
        out_path = f'data/voice_clone/cloned_{acc}.wav'
        base_speaker_tts.tts_to_file(text, spk_id, base_path, speed=1.0)

        source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/{se_file}', map_location=DEVICE)
        converter.convert(audio_src_path=base_path, src_se=source_se,
                           tgt_se=target_se, output_path=out_path, tau=0.3)
        print(f'[{acc}] "{text}" -> {out_path}')
        out_paths.append(out_path)
    return out_paths


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='A6 Speech Processing — run.py')
    parser.add_argument('--model', required=True,
                         choices=['ctc', 'wav2vec2-probe', 'voice-clone'])

    # ctc args
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train', action='store_true')

    # wav2vec2-probe args
    parser.add_argument('--dataset', default='speechcommands')
    parser.add_argument('--classes', default='yes,no,stop,go')

    # voice-clone args
    parser.add_argument('--extract-se', action='store_true')
    parser.add_argument('--reference', default='data/voice_clone/my_voice.mp3')
    parser.add_argument('--accent', default=None, help="us | br | india | au | all")
    parser.add_argument('--language', default=None, help="es | fr | en | zh | jp | kr")
    parser.add_argument('--text', default="Hello, this is a test.")
    parser.add_argument('--generate', action='store_true')

    args = parser.parse_args()

    if args.model == 'ctc':
        if args.train:
            train_ctc(epochs=args.epochs)

    elif args.model == 'wav2vec2-probe':
        if args.train:
            classes = args.classes.split(',')
            run_wav2vec2_probe(classes)

    elif args.model == 'voice-clone':
        if args.extract_se:
            extract_se(args.reference)
        elif args.generate:
            generate_voice(args.text, accent=args.accent, language=args.language)
        else:
            print('Specify --extract-se or --generate for --model voice-clone')


if __name__ == '__main__':
    main()
