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

PANEL_WIDTH = 360


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# LOAD CALIBRATION
# =========================================================

if not os.path.exists(CALIBRATION_FILE):

    print()
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

    print()
    print("ERROR: Homography matrix not found.")
    print("Available keys:", calibration.files)
    exit()


H = np.array(
    H,
    dtype=np.float32
)


print("Homography loaded successfully.")


# =========================================================
# LOAD WORKSPACE POINTS
# =========================================================

if "image_points" in calibration.files:

    workspace_points = calibration["image_points"]

elif "image_pts" in calibration.files:

    workspace_points = calibration["image_pts"]

else:

    print()
    print("ERROR: Workspace image points not found.")
    print("Available keys:", calibration.files)
    exit()


workspace_points = np.array(
    workspace_points,
    dtype=np.int32
).reshape(-1, 2)


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

    print()
    print("ERROR: Could not open camera.")
    exit()


print()
print("Camera opened successfully.")
print()
print("========================================")
print("YOLO + TARGET + HOMOGRAPHY + MATCHING")
print("========================================")
print("Press Q to quit.")


# =========================================================
# COLOR MASK FOR OBJECT SHAPE DETECTION
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
# SHAPE + CENTROID FOR YOLO OBJECT
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
# DETECT BLUE TARGET
# =========================================================

def detect_blue_target(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # -----------------------------------------------------
    # LIGHT BLUE / CYAN TARGET
    #
    # The target in your current setup is light blue/cyan.
    # This is intentionally different from the dark-blue
    # object threshold used by YOLO/object processing.
    # -----------------------------------------------------

    lower_target = np.array(
        [80, 20, 150]
    )

    upper_target = np.array(
        [110, 200, 255]
    )


    mask = cv2.inRange(
        hsv,
        lower_target,
        upper_target
    )


    # -----------------------------------------------------
    # CLEAN MASK
    # -----------------------------------------------------

    kernel = np.ones(
        (7, 7),
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


    # -----------------------------------------------------
    # FIND CONTOURS
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        # Ignore tiny regions

        if area < 1000:

            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        if w <= 0 or h <= 0:

            continue


        aspect_ratio = w / float(h)


        # Target should be approximately square/rectangular

        if 0.65 <= aspect_ratio <= 1.35:

            candidates.append(
                (
                    area,
                    contour,
                    x,
                    y,
                    w,
                    h
                )
            )


    # -----------------------------------------------------
    # NO TARGET FOUND
    # -----------------------------------------------------

    if len(candidates) == 0:

        return None


    # -----------------------------------------------------
    # SELECT LARGEST VALID TARGET
    # -----------------------------------------------------

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )


    area, contour, x, y, w, h = candidates[0]


    # -----------------------------------------------------
    # TARGET CENTROID
    # -----------------------------------------------------

    moments = cv2.moments(
        contour
    )


    if moments["m00"] == 0:

        cx = x + w // 2
        cy = y + h // 2

    else:

        cx = int(
            moments["m10"]
            / moments["m00"]
        )

        cy = int(
            moments["m01"]
            / moments["m00"]
        )


    return {
        "cx": cx,
        "cy": cy,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "contour": contour
    }


# =========================================================
# PIXEL → WORLD COORDINATES
# =========================================================

def pixel_to_world(cx, cy):

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


    return world_x, world_y


# =========================================================
# DRAW INFORMATION PANEL
# =========================================================

def draw_panel(
    panel,
    detections,
    blue_target
):

    panel[:] = 35


    # =====================================================
    # TITLE
    # =====================================================

    cv2.putText(
        panel,
        "OBJECT DETECTIONS",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.line(
        panel,
        (15, 47),
        (PANEL_WIDTH - 15, 47),
        (150, 150, 150),
        1
    )


    # =====================================================
    # OBJECT INFORMATION
    # =====================================================

    y = 78


    for detection in detections:

        color = detection["color"]

        confidence = detection["confidence"]

        shape = detection["shape"]

        cx = detection["cx"]

        cy = detection["cy"]

        world_x = detection["world_x"]

        world_y = detection["world_y"]


        cv2.putText(
            panel,
            f"{color}  {confidence:.2f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )


        cv2.putText(
            panel,
            f"Shape: {shape}",
            (20, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )


        cv2.putText(
            panel,
            f"Center: ({cx}, {cy})",
            (20, y + 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )


        cv2.putText(
            panel,
            f"X: {world_x:.1f} mm",
            (20, y + 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )


        cv2.putText(
            panel,
            f"Y: {world_y:.1f} mm",
            (20, y + 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )


        cv2.line(
            panel,
            (15, y + 96),
            (PANEL_WIDTH - 15, y + 96),
            (80, 80, 80),
            1
        )


        y += 108


    # =====================================================
    # TARGET SECTION
    # =====================================================

    target_y = 410


    cv2.putText(
        panel,
        "BLUE TARGET",
        (20, target_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )


    if blue_target is None:

        cv2.putText(
            panel,
            "Target not detected",
            (20, target_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1
        )

    else:

        target_x = blue_target["world_x"]
        target_y_world = blue_target["world_y"]

        cv2.putText(
            panel,
            f"X: {target_x:.1f} mm",
            (20, target_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )

        cv2.putText(
            panel,
            f"Y: {target_y_world:.1f} mm",
            (20, target_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()


    if not success:

        print("Could not read camera frame.")
        break


    # =====================================================
    # YOLO OBJECT DETECTION
    # =====================================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    result = results[0]


    display_frame = frame.copy()


    # =====================================================
    # DRAW CALIBRATED WORKSPACE
    # =====================================================

    cv2.polylines(
        display_frame,
        [workspace_points],
        True,
        (255, 0, 255),
        2
    )


    # =====================================================
    # DETECTIONS LIST
    # =====================================================

    detections = []


    # =====================================================
    # PROCESS YOLO OBJECTS
    # =====================================================

    if result.boxes is not None:

        for box in result.boxes:

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
            # CLASS
            # -------------------------------------------------

            class_id = int(
                box.cls[0].item()
            )


            class_name = model.names[
                class_id
            ]


            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            confidence = float(
                box.conf[0].item()
            )


            # -------------------------------------------------
            # SHAPE + CENTROID
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
            # FALLBACK CENTROID
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


            # -------------------------------------------------
            # HOMOGRAPHY
            # -------------------------------------------------

            world_x, world_y = (
                pixel_to_world(
                    cx,
                    cy
                )
            )


            # -------------------------------------------------
            # STORE DETECTION
            # -------------------------------------------------

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


            # -------------------------------------------------
            # DRAW YOLO BOX
            # -------------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )


            # -------------------------------------------------
            # DRAW CENTROID
            # -------------------------------------------------

            cv2.circle(
                display_frame,
                (cx, cy),
                6,
                (0, 0, 255),
                -1
            )


    # =====================================================
    # DETECT BLUE TARGET
    # =====================================================

    blue_target = detect_blue_target(
        frame
    )


    # =====================================================
    # PROCESS BLUE TARGET
    # =====================================================

    if blue_target is not None:

        target_cx = blue_target["cx"]
        target_cy = blue_target["cy"]


        target_world_x, target_world_y = (
            pixel_to_world(
                target_cx,
                target_cy
            )
        )


        blue_target["world_x"] = (
            target_world_x
        )

        blue_target["world_y"] = (
            target_world_y
        )


        # -------------------------------------------------
        # DRAW TARGET BOUNDING RECTANGLE
        # -------------------------------------------------

        tx = blue_target["x"]
        ty = blue_target["y"]
        tw = blue_target["w"]
        th = blue_target["h"]


        cv2.rectangle(
            display_frame,
            (tx, ty),
            (tx + tw, ty + th),
            (0, 255, 255),
            3
        )


        # -------------------------------------------------
        # DRAW TARGET CENTROID
        # -------------------------------------------------

        cv2.circle(
            display_frame,
            (target_cx, target_cy),
            7,
            (0, 255, 255),
            -1
        )


        # -------------------------------------------------
        # TARGET LABEL
        # -------------------------------------------------

        cv2.putText(
            display_frame,
            "BLUE TARGET",
            (tx, max(20, ty - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


    # =====================================================
    # DRAW PANEL
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
        detections,
        blue_target
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
        "YOLO + Target + Homography + Matching",
        combined
    )


    # =====================================================
    # TERMINAL MATCHING OUTPUT
    # =====================================================

    # Find BLUE object

    blue_object = None


    for detection in detections:

        if detection["color"] == "BLUE":

            blue_object = detection

            break


    # -----------------------------------------------------
    # PRINT MATCHING INFORMATION
    # -----------------------------------------------------

    if blue_object is not None and blue_target is not None:

        print()
        print("========================================")
        print("COLOR MATCHING")
        print("========================================")

        print(
            "BLUE OBJECT → BLUE TARGET"
        )

        print()

        print(
            f"Pick position:"
        )

        print(
            f"X = "
            f"{blue_object['world_x']:.1f} mm"
        )

        print(
            f"Y = "
            f"{blue_object['world_y']:.1f} mm"
        )

        print()

        print(
            f"Place position:"
        )

        print(
            f"X = "
            f"{blue_target['world_x']:.1f} mm"
        )

        print(
            f"Y = "
            f"{blue_target['world_y']:.1f} mm"
        )

        print(
            "========================================"
        )


    elif blue_object is not None:

        print(
            "BLUE OBJECT detected, "
            "but BLUE TARGET not detected."
        )


    elif blue_target is not None:

        print(
            "BLUE TARGET detected, "
            "but BLUE OBJECT not detected."
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
print("YOLO + TARGET + HOMOGRAPHY + MATCHING")
print("FINISHED")
print("========================================")