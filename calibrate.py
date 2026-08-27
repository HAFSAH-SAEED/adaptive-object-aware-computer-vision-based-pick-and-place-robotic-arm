
import cv2
import numpy as np

# Open iPhone camera through DroidCam
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open iPhone camera.")
    exit()

print("iPhone camera opened successfully!")
print()
print("Click the 4 workspace corners in this order:")
print("1. TOP-LEFT")
print("2. TOP-RIGHT")
print("3. BOTTOM-RIGHT")
print("4. BOTTOM-LEFT")
print()
print("Press Q to quit.")

# Workspace dimensions
WORKSPACE_LENGTH = 300.0   # mm
WORKSPACE_WIDTH = 243.0    # mm

# Real-world coordinates of the four corners
real_points = np.array([
    [0, 0],
    [300, 0],
    [300, 243],
    [0, 243]
], dtype=np.float32)

# Store image coordinates
image_points = []


# Mouse function
def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(image_points) < 4:

            image_points.append([x, y])

            print(
                f"Point {len(image_points)}: ({x}, {y})"
            )


# Create window
cv2.namedWindow("Workspace Calibration")

cv2.setMouseCallback(
    "Workspace Calibration",
    mouse_callback
)


# Camera loop
while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    display = frame.copy()

    # Draw selected points
    for i, point in enumerate(image_points):

        x, y = point

        cv2.circle(
            display,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            f"P{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Draw lines
    if len(image_points) >= 2:

        points = np.array(
            image_points,
            dtype=np.int32
        )

        for i in range(len(points) - 1):

            cv2.line(
                display,
                tuple(points[i]),
                tuple(points[i + 1]),
                (0, 255, 0),
                2
            )

    # Close the rectangle after 4 points
    if len(image_points) == 4:

        cv2.line(
            display,
            tuple(image_points[3]),
            tuple(image_points[0]),
            (0, 255, 0),
            2
        )

    # Display number of selected points
    cv2.putText(
        display,
        f"Points selected: {len(image_points)}/4",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Workspace Calibration",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# Calculate homography
if len(image_points) == 4:

    image_points_np = np.array(
        image_points,
        dtype=np.float32
    )

    H, status = cv2.findHomography(
        image_points_np,
        real_points
    )

    print()
    print("================================")
    print("HOMOGRAPHY CALIBRATION")
    print("================================")

    print()
    print("Image points:")
    print(image_points_np)

    print()
    print("Real-world points (mm):")
    print(real_points)

    print()
    print("Homography matrix:")
    print(H)

    # Save calibration data
    np.savez(
        "calibration_data.npz",
        homography=H,
        image_points=image_points_np,
        real_points=real_points
    )

    print()
    print("Calibration saved successfully!")
    print("File: calibration_data.npz")

else:

    print()
    print("Calibration cancelled.")
    print("Exactly 4 points are required.")


# Close camera
camera.release()

cv2.destroyAllWindows()