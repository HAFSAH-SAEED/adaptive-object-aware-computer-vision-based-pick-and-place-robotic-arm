import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================

CAMERA_ID = 1

IMAGE_DIR = "yolo_dataset/images"
LABEL_DIR = "yolo_dataset/labels"

MIN_AREA = 300


# =========================================================
# CREATE DATASET FOLDERS
# =========================================================

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


# =========================================================
# OPEN CAMERA
# =========================================================

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    print("Could not open iPhone camera.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("iPhone camera opened successfully!")


# =========================================================
# CURRENT CALIBRATED WORKSPACE
# 640 x 480 camera
#
# These are the SAME points selected in calibrate.py
# =========================================================

workspace_points = np.array([
    [54, 86],      # P1 - TOP LEFT
    [266, 89],     # P2 - TOP RIGHT
    [266, 343],    # P3 - BOTTOM RIGHT
    [59, 345]      # P4 - BOTTOM LEFT
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


# GREEN
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
# CHECK IF POINT IS INSIDE WORKSPACE
# =========================================================

def inside_workspace(cx, cy):

    result = cv2.pointPolygonTest(
        workspace_points,
        (float(cx), float(cy)),
        False
    )

    return result >= 0


# =========================================================
# DETECT RED / BLUE / GREEN
# =========================================================

def detect_objects(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # RED
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # BLUE
    # -----------------------------------------------------

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )


    # -----------------------------------------------------
    # GREEN
    # -----------------------------------------------------

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


        # Remove noise
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )


        # Close gaps
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


        valid_contours = []


        # =================================================
        # FILTER CONTOURS
        # =================================================

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


            # Only accept objects inside workspace
            if not inside_workspace(
                cx,
                cy
            ):
                continue


            valid_contours.append(
                contour
            )


        # =================================================
        # KEEP LARGEST OBJECT OF THIS COLOR
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


            # YOLO normalized center
            center_x = (
                x + w / 2
            ) / image_width

            center_y = (
                y + h / 2
            ) / image_height


            # YOLO normalized width/height
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


print()
print("========================================")
print("MANUAL YOLO DATASET COLLECTION")
print("========================================")
print()
print("RED   = Class 0")
print("BLUE  = Class 1")
print("GREEN = Class 2")
print()
print("SPACE = Capture image")
print("Q     = Quit")
print()
print("Move objects to a new position.")
print("Remove your hand.")
print("Press SPACE when ready.")
print()


# =========================================================
# CAMERA LOOP
# =========================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "Could not read camera frame."
        )

        break


    # =====================================================
    # DETECT OBJECTS
    # =====================================================

    detections = detect_objects(
        frame
    )


    # =====================================================
    # DISPLAY COPY
    #
    # Original frame stays untouched.
    # =====================================================

    display_frame = frame.copy()


    # =====================================================
    # DRAW WORKSPACE
    # =====================================================

    cv2.polylines(
        display_frame,
        [workspace_points],
        True,
        (255, 0, 255),
        2
    )


    # =====================================================
    # DRAW OBJECTS
    # =====================================================

    for detection in detections:

        x = detection["x"]
        y = detection["y"]
        w = detection["w"]
        h = detection["h"]

        cx = detection["cx"]
        cy = detection["cy"]

        color_name = detection["color"]


        # Bounding box
        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Centroid
        cv2.circle(
            display_frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )


        # Object name
        cv2.putText(
            display_frame,
            color_name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )


    # =====================================================
    # INFORMATION
    # =====================================================

    cv2.putText(
        display_frame,
        f"Objects detected: {len(detections)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )


    cv2.putText(
        display_frame,
        f"Images saved: {image_count}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


    cv2.putText(
        display_frame,
        "SPACE = Capture | Q = Quit",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Manual YOLO Dataset Collection",
        display_frame
    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # SPACE = CAPTURE
    # =====================================================

    if key == 32:

        # Need at least one object
        if len(detections) == 0:

            print(
                "No valid objects detected. "
                "Image NOT saved."
            )

            continue


        # -------------------------------------------------
        # IMAGE NUMBER
        # -------------------------------------------------

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


        # -------------------------------------------------
        # SAVE CLEAN IMAGE
        # -------------------------------------------------

        cv2.imwrite(
            image_path,
            frame
        )


        # -------------------------------------------------
        # SAVE YOLO LABEL
        # -------------------------------------------------

        image_height, image_width = (
            frame.shape[:2]
        )


        save_yolo_label(
            label_path,
            detections,
            image_width,
            image_height
        )


        # -------------------------------------------------
        # PRINT RESULT
        # -------------------------------------------------

        detected_colors = [
            detection["color"]
            for detection in detections
        ]


        print(
            f"Captured image {image_count} | "
            f"Objects: {len(detections)} | "
            f"Colors: {detected_colors}"
        )


    # =====================================================
    # Q = QUIT
    # =====================================================

    elif key == ord("q"):

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
print(
    f"Images saved: {image_count}"
)

print(
    f"Images folder: {IMAGE_DIR}"
)

print(
    f"Labels folder: {LABEL_DIR}"
)