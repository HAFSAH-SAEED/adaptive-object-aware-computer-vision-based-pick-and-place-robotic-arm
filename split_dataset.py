
import os
import random
import shutil

# =========================================================
# SETTINGS
# =========================================================

IMAGE_SOURCE = "yolo_dataset/images"
LABEL_SOURCE = "yolo_dataset/labels"

TRAIN_IMAGES = "yolo_dataset/images/train"
VAL_IMAGES = "yolo_dataset/images/val"

TRAIN_LABELS = "yolo_dataset/labels/train"
VAL_LABELS = "yolo_dataset/labels/val"

TRAIN_RATIO = 0.8

random.seed(42)


# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(VAL_IMAGES, exist_ok=True)

os.makedirs(TRAIN_LABELS, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)


# =========================================================
# GET ONLY ORIGINAL IMAGES
# =========================================================

images = [
    f for f in os.listdir(IMAGE_SOURCE)
    if f.lower().endswith(".jpg")
    and os.path.isfile(
        os.path.join(IMAGE_SOURCE, f)
    )
]


print(f"Total images found: {len(images)}")


# =========================================================
# SHUFFLE
# =========================================================

random.shuffle(images)


# =========================================================
# SPLIT 80 / 20
# =========================================================

split_index = int(
    len(images) * TRAIN_RATIO
)

train_images = images[:split_index]
val_images = images[split_index:]


print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")


# =========================================================
# FUNCTION TO MOVE IMAGE + LABEL
# =========================================================

def move_pair(
    image_name,
    destination_images,
    destination_labels
):

    label_name = (
        os.path.splitext(image_name)[0]
        + ".txt"
    )

    image_source = os.path.join(
        IMAGE_SOURCE,
        image_name
    )

    label_source = os.path.join(
        LABEL_SOURCE,
        label_name
    )

    image_destination = os.path.join(
        destination_images,
        image_name
    )

    label_destination = os.path.join(
        destination_labels,
        label_name
    )


    # Move image
    shutil.move(
        image_source,
        image_destination
    )


    # Move matching label
    if os.path.exists(label_source):

        shutil.move(
            label_source,
            label_destination
        )

    else:

        print(
            f"WARNING: Label missing for {image_name}"
        )


# =========================================================
# MOVE TRAINING DATA
# =========================================================

for image_name in train_images:

    move_pair(
        image_name,
        TRAIN_IMAGES,
        TRAIN_LABELS
    )


# =========================================================
# MOVE VALIDATION DATA
# =========================================================

for image_name in val_images:

    move_pair(
        image_name,
        VAL_IMAGES,
        VAL_LABELS
    )


# =========================================================
# FINISHED
# =========================================================

print()
print("========================================")
print("DATASET SPLIT COMPLETE")
print("========================================")

print(
    f"Training images: {len(train_images)}"
)

print(
    f"Validation images: {len(val_images)}"
)

print()
print("Training images:")
print(TRAIN_IMAGES)

print()
print("Validation images:")
print(VAL_IMAGES)