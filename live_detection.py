
import cv2
import numpy as np
import math

# Open the default camera
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("Live blue-object detection started!")
print("Press Q to quit.")

while True:

    # Capture one frame from the camera
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Convert the frame from BGR to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])

    # Create blue mask
    blue_mask = cv2.inRange(
        hsv_image,
        lower_blue,
        upper_blue
    )

    # Create kernel for noise removal
    kernel = np.ones((5, 5), np.uint8)

    # Clean the mask
    clean_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find external contours
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a copy for drawing
    result = frame.copy()

    # Minimum area for a detected object
    minimum_area = 1000

    # Process detected contours
    for contour in contours:

        # Calculate contour area
        area = cv2.contourArea(contour)

        # Ignore small regions
        if area < minimum_area:
            continue

        # Calculate perimeter
        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Calculate circularity
        if perimeter != 0:
            circularity = (
                4 * math.pi * area
            ) / (perimeter * perimeter)
        else:
            circularity = 0

        # Approximate contour
        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # Count contour vertices
        number_of_vertices = len(approximation)

        # Basic shape classification
        if circularity > 0.75:
            shape = "ROUND / CIRCLE"
        elif number_of_vertices == 3:
            shape = "TRIANGLE"
        elif number_of_vertices == 4:
            shape = "RECTANGLE / SQUARE"
        else:
            shape = "IRREGULAR"

        # Calculate bounding box
        x, y, width, height = cv2.boundingRect(
            contour
        )

        # Calculate centroid
        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            centroid_x = int(
                moments["m10"] / moments["m00"]
            )

            centroid_y = int(
                moments["m01"] / moments["m00"]
            )

            # Draw centroid
            cv2.circle(
                result,
                (centroid_x, centroid_y),
                6,
                (0, 0, 255),
                -1
            )

            # Display centroid
            cv2.putText(
                result,
                f"Center: ({centroid_x}, {centroid_y})",
                (x, y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Draw contour
        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            3
        )

        # Draw bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
        )

        # Display shape
        cv2.putText(
            result,
            f"Shape: {shape}",
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Display size
        cv2.putText(
            result,
            f"Size: {width} x {height} px",
            (x, y + height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Show live detection
    cv2.imshow(
        "Live Blue Object Detection",
        result
    )

    # Show mask
    cv2.imshow(
        "Live Blue Mask",
        clean_mask
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()

# Close all windows
cv2.destroyAllWindows()