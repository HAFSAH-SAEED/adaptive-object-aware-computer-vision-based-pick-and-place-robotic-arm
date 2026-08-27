import cv2
import numpy as np

# ==============================
# CAMERA
# ==============================

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open iPhone camera.")
    exit()

print("iPhone camera opened successfully!")


# ==============================
# LOAD HOMOGRAPHY
# ==============================

try:
    calibration = np.load("calibration_data.npz")
    H = calibration["homography"]

    print("Calibration loaded successfully!")

except Exception as e:
    print("Could not load calibration_data.npz")
    print(e)

    camera.release()
    exit()


# ==============================
# BLUE COLOR RANGE
# ==============================

lower_blue = np.array([90, 80, 50])
upper_blue = np.array([140, 255, 255])


# ==============================
# MAIN LOOP
# ==============================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create blue mask
    mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process largest blue object
    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest_contour
        )

        if area > 500:

            # Calculate centroid
            M = cv2.moments(
                largest_contour
            )

            if M["m00"] != 0:

                cx = int(
                    M["m10"] / M["m00"]
                )

                cy = int(
                    M["m01"] / M["m00"]
                )

                # ==============================
                # PIXEL → WORLD COORDINATES
                # ==============================

                pixel_point = np.array(
                    [[[cx, cy]]],
                    dtype=np.float32
                )

                world_point = cv2.perspectiveTransform(
                    pixel_point,
                    H
                )

                X = world_point[0][0][0]
                Y = world_point[0][0][1]

                # ==============================
                # DRAW CONTOUR
                # ==============================

                cv2.drawContours(
                    frame,
                    [largest_contour],
                    -1,
                    (0, 255, 0),
                    2
                )

                # ==============================
                # DRAW CENTROID
                # ==============================

                cv2.circle(
                    frame,
                    (cx, cy),
                    6,
                    (0, 0, 255),
                    -1
                )

                # ==============================
                # PIXEL COORDINATES - BLACK
                # ==============================

                cv2.putText(
                    frame,
                    f"Pixel: ({cx}, {cy})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

                # ==============================
                # WORLD COORDINATES - BLACK
                # ==============================

                cv2.putText(
                    frame,
                    f"World: X={X:.1f} mm, Y={Y:.1f} mm",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

    # ==============================
    # SHOW CAMERA
    # ==============================

    cv2.imshow(
        "Coordinate Mapping",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

camera.release()
cv2.destroyAllWindows()

print("Coordinate mapping stopped.")