import os
import json
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split


# Change these paths if needed
COCO_IMAGES_DIR = Path(r"C:\Users\svrat\fiftyone\coco-2017\validation\data")
COCO_ANN_FILE = Path(r"C:\Users\svrat\fiftyone\coco-2017\raw\instances_val2017.json")

PROJECT_DATA_DIR = Path("data")

# Use small number first so training is faster
MAX_IMAGES = 200
VALID_RATIO = 0.2


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox format:
    x_min, y_min, width, height

    YOLO bbox format:
    center_x, center_y, width, height
    normalized between 0 and 1
    """
    x, y, w, h = bbox

    center_x = (x + w / 2) / img_w
    center_y = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    return center_x, center_y, norm_w, norm_h


def main():
    print("Loading COCO annotations...")

    with open(COCO_ANN_FILE, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # COCO category id is not continuous, so convert it to 0-based class index
    categories = sorted(categories, key=lambda x: x["id"])
    cat_id_to_class_idx = {
        cat["id"]: idx for idx, cat in enumerate(categories)
    }

    class_names = [cat["name"] for cat in categories]

    # Group annotations by image_id
    image_id_to_anns = {}

    for ann in annotations:
        image_id = ann["image_id"]

        if ann.get("iscrowd", 0) == 1:
            continue

        if image_id not in image_id_to_anns:
            image_id_to_anns[image_id] = []

        image_id_to_anns[image_id].append(ann)

    # Keep only images that have at least one annotation
    images_with_annotations = [
        img for img in images if img["id"] in image_id_to_anns
    ]

    # Use small subset first
    images_with_annotations = images_with_annotations[:MAX_IMAGES]

    train_imgs, valid_imgs = train_test_split(
        images_with_annotations,
        test_size=VALID_RATIO,
        random_state=42,
    )

    # Create folders
    for split in ["train", "valid"]:
        (PROJECT_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROJECT_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Write class names
    with open(PROJECT_DATA_DIR / "classes.names", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    def process_split(split_name, split_images):
        txt_path = PROJECT_DATA_DIR / f"{split_name}.txt"

        with open(txt_path, "w") as txt_file:
            for img in split_images:
                img_id = img["id"]
                filename = img["file_name"]

                src_img_path = COCO_IMAGES_DIR / filename
                dst_img_path = PROJECT_DATA_DIR / "images" / split_name / filename

                if not src_img_path.exists():
                    print("Missing image:", src_img_path)
                    continue

                shutil.copy(src_img_path, dst_img_path)

                img_w = img["width"]
                img_h = img["height"]

                label_filename = Path(filename).stem + ".txt"
                label_path = PROJECT_DATA_DIR / "labels" / split_name / label_filename

                anns = image_id_to_anns[img_id]

                with open(label_path, "w") as label_file:
                    for ann in anns:
                        class_idx = cat_id_to_class_idx[ann["category_id"]]
                        yolo_box = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)

                        line = (
                            f"{class_idx} "
                            f"{yolo_box[0]:.6f} "
                            f"{yolo_box[1]:.6f} "
                            f"{yolo_box[2]:.6f} "
                            f"{yolo_box[3]:.6f}\n"
                        )

                        label_file.write(line)

                # Important: write image path using forward slashes
                txt_file.write(str(dst_img_path).replace("\\", "/") + "\n")

    process_split("train", train_imgs)
    process_split("valid", valid_imgs)

    # Create custom.data
    with open(PROJECT_DATA_DIR / "custom.data", "w") as f:
        f.write(f"classes={len(class_names)}\n")
        f.write("train=data/train.txt\n")
        f.write("valid=data/valid.txt\n")
        f.write("names=data/classes.names\n")
        f.write("backup=outputs/checkpoints\n")

    print("Done.")
    print(f"Classes: {len(class_names)}")
    print(f"Train images: {len(train_imgs)}")
    print(f"Valid images: {len(valid_imgs)}")
    print("Created:")
    print("data/train.txt")
    print("data/valid.txt")
    print("data/classes.names")
    print("data/custom.data")


if __name__ == "__main__":
    main()