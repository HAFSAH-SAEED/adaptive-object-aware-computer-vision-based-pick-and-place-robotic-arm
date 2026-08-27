import cv2
import numpy as np
import math

# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open iPhone camera.")
    exit()

print("iPhone camera opened successfully!")


# =========================================================
# LOAD HOMOGRAPHY
# =========================================================

try:
    calibration = np.load("calibration_data.npz")
    H = calibration["homography"]

    print("Calibration loaded successfully!")

except Exception as e:
    print("Could not load calibration_data.npz")
    print(e)
    camera.release()
    exit()


# =========================================================
# WORKSPACE CORNERS
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
# SETTINGS
# =========================================================

MIN_AREA = 300


# =========================================================
# SHAPE DETECTION
# =========================================================

def detect_shape(contour):

    area = cv2.contourArea(contour)

    if area <= 0:
        return "Unknown"

    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return "Unknown"

    circularity = (
        4 * np.pi * area
        / (perimeter * perimeter)
    )

    if circularity > 0.65:
        return "Circle"

    approximation = cv2.approxPolyDP(
        contour,
        0.03 * perimeter,
        True
    )

    vertices = len(approximation)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:

        x, y, w, h = cv2.boundingRect(
            approximation
        )

        aspect_ratio = w / float(h)

        if 0.75 <= aspect_ratio <= 1.25:
            return "Square"

        return "Rectangle"

    elif vertices > 5:
        return "Circle"

    return "Unknown"


# =========================================================
# PIXEL TO WORLD
# =========================================================

def pixel_to_world(cx, cy):

    point = np.array(
        [[[cx, cy]]],
        dtype=np.float32
    )

    world = cv2.perspectiveTransform(
        point,
        H
    )

    X = float(world[0][0][0])
    Y = float(world[0][0][1])

    return X, Y


# =========================================================
# REAL WORLD SIZE
# =========================================================

def calculate_world_size(x, y, w, h):

    points = np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]]
    ], dtype=np.float32)

    world_points = cv2.perspectiveTransform(
        points,
        H
    )

    p1 = world_points[0][0]
    p2 = world_points[1][0]
    p4 = world_points[3][0]

    width_mm = math.dist(p1, p2)
    height_mm = math.dist(p1, p4)

    return width_mm, height_mm


# =========================================================
# WORKSPACE CHECK
# =========================================================

def inside_workspace(cx, cy):

    result = cv2.pointPolygonTest(
        workspace_points,
        (float(cx), float(cy)),
        False
    )

    return result >= 0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break


    # =====================================================
    # HSV
    # =====================================================

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # =====================================================
    # COLOR MASKS
    # =====================================================

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

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

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


    # =====================================================
    # OBJECT LIST
    # =====================================================

    detected_objects = []


    # =====================================================
    # DETECTION
    # =====================================================

    for color_name, mask in masks.items():

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


        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_AREA:
                continue


            # Bounding box
            x, y, w, h = cv2.boundingRect(
                contour
            )


            # Centroid
            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(
                M["m10"] / M["m00"]
            )

            cy = int(
                M["m01"] / M["m00"]
            )


            # Workspace filter
            if not inside_workspace(cx, cy):
                continue


            # World coordinates
            X, Y = pixel_to_world(
                cx,
                cy
            )


            # Physical limits
            if X < 0 or X > 300:
                continue

            if Y < 0 or Y > 243:
                continue


            # Shape
            shape = detect_shape(contour)


            # Real size
            width_mm, height_mm = calculate_world_size(
                x,
                y,
                w,
                h
            )


            detected_objects.append({
                "color": color_name,
                "shape": shape,
                "cx": cx,
                "cy": cy,
                "X": X,
                "Y": Y,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "contour": contour
            })


    # =====================================================
    # DRAW WORKSPACE
    # =====================================================

    cv2.polylines(
        frame,
        [workspace_points],
        True,
        (255, 0, 255),
        2
    )


    # =====================================================
    # DRAW OBJECTS ON CAMERA
    # =====================================================

    for obj in detected_objects:

        contour = obj["contour"]

        x = obj["x"]
        y = obj["y"]
        w = obj["w"]
        h = obj["h"]

        cx = obj["cx"]
        cy = obj["cy"]


        # Contour
        cv2.drawContours(
            frame,
            [contour],
            -1,
            (0, 255, 0),
            2
        )


        # Bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )


        # Centroid
        cv2.circle(
            frame,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )


    # =====================================================
    # CREATE RIGHT INFORMATION PANEL
    # =====================================================

    panel_width = 430

    height = frame.shape[0]

    panel = np.ones(
        (height, panel_width, 3),
        dtype=np.uint8
    ) * 255


    # =====================================================
    # PANEL TITLE
    # =====================================================

    cv2.putText(
        panel,
        "VISION SYSTEM",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )


    cv2.putText(
        panel,
        f"Objects detected: {len(detected_objects)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )


    # =====================================================
    # OBJECT INFORMATION
    # =====================================================

    for i, obj in enumerate(detected_objects):

        start_y = 120 + (i * 115)

        color = obj["color"]
        shape = obj["shape"]

        cx = obj["cx"]
        cy = obj["cy"]

        X = obj["X"]
        Y = obj["Y"]

        width_mm = obj["width_mm"]
        height_mm = obj["height_mm"]


        # Object heading
        cv2.putText(
            panel,
            f"{i + 1}. {color} | {shape}",
            (20, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )


        # Centroid
        cv2.putText(
            panel,
            f"Center: ({cx}, {cy})",
            (20, start_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1
        )


        # World coordinates
        cv2.putText(
            panel,
            f"World: X={X:.1f} Y={Y:.1f} mm",
            (20, start_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1
        )


        # Real size
        cv2.putText(
            panel,
            f"Size: {width_mm:.1f} x {height_mm:.1f} mm",
            (20, start_y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1
        )


    # =====================================================
    # COMBINE CAMERA + PANEL
    # =====================================================

    combined = np.hstack(
        (frame, panel)
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Vision Guided Object Detection",
        combined
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

print("Multi-color detection stopped.")