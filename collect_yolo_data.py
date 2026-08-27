import cv2
import numpy as np
import os
import time

# =========================================================
# SETTINGS
# =========================================================

CAMERA_ID = 1

IMAGE_DIR = "yolo_dataset/images"
LABEL_DIR = "yolo_dataset/labels"

MIN_AREA = 300

SAVE_INTERVAL = 1.0


# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    print("Could not open iPhone camera.")
    exit()

print("iPhone camera opened successfully!")


# =========================================================
# WORKSPACE
# =========================================================

workspace_points = np.array([
    [79, 117],
    [284, 114],
    [287, 376],
    [83, 369]
], dtype=np.int32)


# =========================================================
# HSV COLOR RANGES
# =========================================================

# RED
lower_red_1 = np.array([0, 100, 80])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 100, 80])
upper_red_2 = np.array([179, 255, 255])


# BLUE
lower_blue = np.array([90, 80, 50])
upper_blue = np.array([140, 255, 255])


# LIGHT GREEN
lower_green = np.array([35, 35, 140])
upper_green = np.array([90, 255, 255])


# =========================================================
# YOLO CLASS IDs
# =========================================================

CLASS_IDS = {
    "RED": 0,
    "BLUE": 1,
    "GREEN": 2
}


# =========================================================
# CHECK WORKSPACE
# =========================================================

def inside_workspace(cx, cy):

    result = cv2.pointPolygonTest(
        workspace_points,
        (float(cx), float(cy)),
        False
    )

    return result >= 0


# =========================================================
# DETECT ONE OBJECT PER COLOR
# =========================================================

def detect_objects(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # =====================================================
    # COLOR MASKS
    # =====================================================

    # RED

    red_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    red_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    red_mask = cv2.bitwise_or(
        red_1,
        red_2
    )


    # BLUE

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )


    # GREEN

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )


    masks = {
        "RED": red_mask,
        "BLUE": blue_mask,
        "GREEN": green_mask
    }


    detections = []


    # =====================================================
    # PROCESS EACH COLOR
    # =====================================================

    for color_name, mask in masks.items():

        kernel = np.ones(
            (5, 5),
            np.uint8
        )


        # Clean noise
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # =================================================
        # KEEP ONLY VALID CONTOURS
        # =================================================

        valid_contours = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < MIN_AREA:
                continue


            M = cv2.moments(
                contour
            )

            if M["m00"] == 0:
                continue


            cx = int(
                M["m10"] / M["m00"]
            )

            cy = int(
                M["m01"] / M["m00"]
            )


            # Only keep objects inside workspace
            if not inside_workspace(
                cx,
                cy
            ):
                continue


            valid_contours.append(
                contour
            )


        # =================================================
        # KEEP ONLY LARGEST OBJECT
        # =================================================

        if len(valid_contours) == 0:
            continue


        largest_contour = max(
            valid_contours,
            key=cv2.contourArea
        )


        # Bounding box

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )


        # Centroid

        M = cv2.moments(
            largest_contour
        )

        cx = int(
            M["m10"] / M["m00"]
        )

        cy = int(
            M["m01"] / M["m00"]
        )


        detections.append({
            "class_id": CLASS_IDS[color_name],
            "color": color_name,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy
        })


    return detections


# =========================================================
# SAVE YOLO LABEL
# =========================================================

def save_yolo_label(
    label_path,
    detections,
    image_width,
    image_height
):

    with open(
        label_path,
        "w"
    ) as file:

        for detection in detections:

            class_id = detection["class_id"]

            x = detection["x"]
            y = detection["y"]
            w = detection["w"]
            h = detection["h"]


            # YOLO normalized center coordinates

            center_x = (
                x + w / 2
            ) / image_width

            center_y = (
                y + h / 2
            ) / image_height


            # YOLO normalized dimensions

            width = (
                w / image_width
            )

            height = (
                h / image_height
            )


            file.write(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


# =========================================================
# MAIN
# =========================================================

image_count = 0

last_save_time = 0


print()
print("========================================")
print("CLEAN YOLO DATASET COLLECTION")
print("========================================")
print()
print("Classes:")
print("0 = RED")
print("1 = BLUE")
print("2 = GREEN")
print()
print("Only the largest valid object")
print("of each color will be labelled.")
print()
print("Start with 5 test images.")
print()
print("Press Q to quit.")
print()


# =========================================================
# LOOP
# =========================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break


    # =====================================================
    # DETECT
    # =====================================================

    detections = detect_objects(
        frame
    )


    # =====================================================
    # SAVE CLEAN ORIGINAL FRAME
    #
    # IMPORTANT:
    # We save BEFORE drawing boxes/text.
    # =====================================================

    current_time = time.time()

    if (
        current_time - last_save_time
        >= SAVE_INTERVAL
    ):

        if len(detections) > 0:

            image_count += 1


            image_name = (
                f"image_{image_count:04d}.jpg"
            )

            label_name = (
                f"image_{image_count:04d}.txt"
            )


            image_path = os.path.join(
                IMAGE_DIR,
                image_name
            )

            label_path = os.path.join(
                LABEL_DIR,
                label_name
            )


            # ---------------------------------------------
            # SAVE CLEAN IMAGE
            # ---------------------------------------------

            cv2.imwrite(
                image_path,
                frame
            )


            # ---------------------------------------------
            # SAVE LABELS
            # ---------------------------------------------

            image_height, image_width = (
                frame.shape[:2]
            )


            save_yolo_label(
                label_path,
                detections,
                image_width,
                image_height
            )


            print(
                f"Saved image {image_count} | "
                f"Objects: {len(detections)}"
            )


            last_save_time = current_time


    # =====================================================
    # DRAW WORKSPACE FOR DISPLAY ONLY
    # =====================================================

    cv2.polylines(
        frame,
        [workspace_points],
        True,
        (255, 0, 255),
        2
    )


    # =====================================================
    # DRAW DETECTIONS FOR DISPLAY ONLY
    # =====================================================

    for detection in detections:

        x = detection["x"]
        y = detection["y"]
        w = detection["w"]
        h = detection["h"]

        cx = detection["cx"]
        cy = detection["cy"]

        color_name = detection["color"]


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            frame,
            color_name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )


    # =====================================================
    # DISPLAY INFO
    # =====================================================

    cv2.putText(
        frame,
        f"Images saved: {image_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2
    )


    cv2.putText(
        frame,
        "Move objects to different positions",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )


    cv2.putText(
        frame,
        "Q = Quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Clean YOLO Dataset Collection",
        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("DATASET COLLECTION FINISHED")
print("========================================")
print(f"Images saved: {image_count}")
print(f"Images folder: {IMAGE_DIR}")
print(f"Labels folder: {LABEL_DIR}")