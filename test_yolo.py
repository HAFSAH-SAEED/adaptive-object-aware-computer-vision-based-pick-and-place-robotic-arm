from ultralytics import YOLO
import cv2
import numpy as np
import os


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/best.pt"
CALIBRATION_FILE = "calibration_data.npz"

CAMERA_ID = 1
CONFIDENCE = 0.5

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PANEL_WIDTH = 330


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# LOAD LATEST CALIBRATION
# =========================================================

if not os.path.exists(CALIBRATION_FILE):

    print("ERROR: calibration_data.npz not found.")
    print("Run calibrate.py first.")
    exit()


calibration = np.load(CALIBRATION_FILE)

print()
print("Calibration file loaded successfully.")
print("Available data:", calibration.files)


# =========================================================
# LOAD HOMOGRAPHY
# =========================================================

if "homography" in calibration.files:

    H = calibration["homography"]

elif "H" in calibration.files:

    H = calibration["H"]

else:

    print("ERROR: Homography matrix not found.")
    print("Available keys:", calibration.files)
    exit()


H = np.array(
    H,
    dtype=np.float32
)


print("Homography loaded successfully.")


# =========================================================
# LOAD WORKSPACE IMAGE POINTS
# =========================================================

if "image_points" in calibration.files:

    workspace_points = calibration["image_points"]

elif "image_pts" in calibration.files:

    workspace_points = calibration["image_pts"]

else:

    print("ERROR: Workspace points not found.")
    print("Available keys:", calibration.files)
    exit()


workspace_points = np.array(
    workspace_points,
    dtype=np.int32
).reshape(-1, 2)


print()
print("Workspace points loaded:")
print(workspace_points)


# =========================================================
# OPEN CAMERA
# =========================================================

camera = cv2.VideoCapture(CAMERA_ID)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not camera.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print()
print("Camera opened successfully.")
print()
print("========================================")
print("YOLO + SHAPE + HOMOGRAPHY")
print("========================================")
print("Press Q to quit.")


# =========================================================
# COLOR MASK
# =========================================================

def get_color_mask(hsv, color_name):

    if color_name == "RED":

        lower1 = np.array(
            [0, 100, 80]
        )

        upper1 = np.array(
            [10, 255, 255]
        )

        lower2 = np.array(
            [170, 100, 80]
        )

        upper2 = np.array(
            [179, 255, 255]
        )

        mask1 = cv2.inRange(
            hsv,
            lower1,
            upper1
        )

        mask2 = cv2.inRange(
            hsv,
            lower2,
            upper2
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2
        )


    elif color_name == "BLUE":

        lower = np.array(
            [90, 80, 50]
        )

        upper = np.array(
            [140, 255, 255]
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )


    elif color_name == "GREEN":

        lower = np.array(
            [35, 35, 100]
        )

        upper = np.array(
            [90, 255, 255]
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )


    else:

        mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )


    return mask


# =========================================================
# SHAPE + CENTROID
# =========================================================

def detect_shape_and_centroid(
    frame,
    x1,
    y1,
    x2,
    y2,
    color_name
):

    height, width = frame.shape[:2]


    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)


    crop = frame[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        return "UNKNOWN", None


    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )


    mask = get_color_mask(
        hsv,
        color_name
    )


    kernel = np.ones(
        (5, 5),
        np.uint8
    )


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


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:

        return "UNKNOWN", None


    contour = max(
        contours,
        key=cv2.contourArea
    )


    area = cv2.contourArea(
        contour
    )


    if area < 50:

        return "UNKNOWN", None


    perimeter = cv2.arcLength(
        contour,
        True
    )


    if perimeter == 0:

        return "UNKNOWN", None


    epsilon = 0.04 * perimeter


    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )


    circularity = (
        4 * np.pi * area
        / (perimeter * perimeter)
    )


    # =====================================================
    # SHAPE CLASSIFICATION
    # =====================================================

    if circularity > 0.70:

        shape = "CIRCLE"

    elif len(approx) == 4:

        shape = "RECTANGLE"

    elif color_name == "GREEN":

        shape = "RECTANGLE"

    else:

        shape = "UNKNOWN"


    # =====================================================
    # CENTROID
    # =====================================================

    moments = cv2.moments(
        contour
    )


    if moments["m00"] != 0:

        local_cx = int(
            moments["m10"]
            / moments["m00"]
        )

        local_cy = int(
            moments["m01"]
            / moments["m00"]
        )

    else:

        local_cx = crop.shape[1] // 2
        local_cy = crop.shape[0] // 2


    cx = x1 + local_cx
    cy = y1 + local_cy


    return shape, (cx, cy)


# =========================================================
# DRAW INFORMATION PANEL
# =========================================================

def draw_panel(panel, detections):

    # Background

    panel[:] = 35


    # Title

    cv2.putText(
        panel,
        "OBJECT DETECTIONS",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Separator

    cv2.line(
        panel,
        (15, 50),
        (PANEL_WIDTH - 15, 50),
        (150, 150, 150),
        1
    )


    if len(detections) == 0:

        cv2.putText(
            panel,
            "No objects detected",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        return


    y = 85


    for detection in detections:

        color = detection["color"]
        confidence = detection["confidence"]
        shape = detection["shape"]
        cx = detection["cx"]
        cy = detection["cy"]
        world_x = detection["world_x"]
        world_y = detection["world_y"]


        # -------------------------------------------------
        # Object heading
        # -------------------------------------------------

        heading = (
            f"{color}   {confidence:.2f}"
        )


        cv2.putText(
            panel,
            heading,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # Shape

        cv2.putText(
            panel,
            f"Shape: {shape}",
            (20, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # Centroid

        cv2.putText(
            panel,
            f"Centroid: ({cx}, {cy})",
            (20, y + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # X

        cv2.putText(
            panel,
            f"X: {world_x:.1f} mm",
            (20, y + 69),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # Y

        cv2.putText(
            panel,
            f"Y: {world_y:.1f} mm",
            (20, y + 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1
        )


        # Separator

        cv2.line(
            panel,
            (15, y + 110),
            (PANEL_WIDTH - 15, y + 110),
            (90, 90, 90),
            1
        )


        y += 130


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()


    if not success:

        print("Could not read camera frame.")
        break


    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    result = results[0]


    # =====================================================
    # DISPLAY FRAME
    # =====================================================

    display_frame = frame.copy()


    # =====================================================
    # DRAW CURRENT CALIBRATED WORKSPACE
    # =====================================================

    cv2.polylines(
        display_frame,
        [workspace_points],
        True,
        (255, 0, 255),
        2
    )


    # =====================================================
    # STORE DETECTIONS
    # =====================================================

    detections = []


    # =====================================================
    # PROCESS YOLO BOXES
    # =====================================================

    if result.boxes is not None:

        for box in result.boxes:

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )


            x1 = int(x1)
            y1 = int(y1)

            x2 = int(x2)
            y2 = int(y2)


            # -------------------------------------------------
            # Class
            # -------------------------------------------------

            class_id = int(
                box.cls[0].item()
            )


            class_name = model.names[
                class_id
            ]


            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            confidence = float(
                box.conf[0].item()
            )


            # -------------------------------------------------
            # Shape + centroid
            # -------------------------------------------------

            shape, centroid = (
                detect_shape_and_centroid(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    class_name
                )
            )


            # -------------------------------------------------
            # Fallback centroid
            # -------------------------------------------------

            if centroid is None:

                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )

            else:

                cx, cy = centroid


            # =================================================
            # HOMOGRAPHY
            # =================================================

            pixel_point = np.array(
                [
                    [
                        [
                            float(cx),
                            float(cy)
                        ]
                    ]
                ],
                dtype=np.float32
            )


            world_point = cv2.perspectiveTransform(
                pixel_point,
                H
            )


            world_x = float(
                world_point[0][0][0]
            )

            world_y = float(
                world_point[0][0][1]
            )


            # =================================================
            # SAVE DETECTION
            # =================================================

            detections.append(
                {
                    "color": class_name,
                    "confidence": confidence,
                    "shape": shape,
                    "cx": cx,
                    "cy": cy,
                    "world_x": world_x,
                    "world_y": world_y
                }
            )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )


            # =================================================
            # DRAW CENTROID
            # =================================================

            cv2.circle(
                display_frame,
                (cx, cy),
                6,
                (0, 0, 255),
                -1
            )


    # =====================================================
    # CREATE INFORMATION PANEL
    # =====================================================

    panel = np.zeros(
        (
            FRAME_HEIGHT,
            PANEL_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    draw_panel(
        panel,
        detections
    )


    # =====================================================
    # COMBINE CAMERA + PANEL
    # =====================================================

    combined = np.hstack(
        (
            display_frame,
            panel
        )
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "YOLO + Shape + Homography",
        combined
    )


    # =====================================================
    # TERMINAL OUTPUT
    # =====================================================

    for detection in detections:

        print(
            f"{detection['color']} | "
            f"confidence="
            f"{detection['confidence']:.2f} | "
            f"shape="
            f"{detection['shape']} | "
            f"centroid="
            f"({detection['cx']},"
            f"{detection['cy']}) | "
            f"world="
            f"({detection['world_x']:.1f}, "
            f"{detection['world_y']:.1f}) mm"
        )


    # =====================================================
    # QUIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("YOLO + SHAPE + HOMOGRAPHY FINISHED")
print("========================================")