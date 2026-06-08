import os
import fiftyone as fo
import fiftyone.zoo as foz

# Download COCO-2017 validation split
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco-2017-validation",
)

dataset.persistent = True

fo_base = os.path.expanduser("~/fiftyone/coco-2017")

path2data = os.path.join(fo_base, "validation", "data")
path2json = os.path.join(fo_base, "raw", "instances_val2017.json")

print("Images:", path2data)
print("Annotations:", path2json)

print("Images exist:", os.path.isdir(path2data))
print("Annotations exist:", os.path.isfile(path2json))