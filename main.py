import cv2
import numpy as np
import math

# Load the image
image = cv2.imread("test_objects.jpeg")

# Check whether the image was loaded
if image is None:
    print("Could not load image.")
    exit()

print("Image loaded successfully!")

# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Blue HSV range
lower_blue = np.array([90, 100, 50])
upper_blue = np.array([130, 255, 255])

# Create blue mask
blue_mask = cv2.inRange(
    hsv_image,
    lower_blue,
    upper_blue
)

# Remove small noise
kernel = np.ones((5, 5), np.uint8)

clean_mask = cv2.morphologyEx(
    blue_mask,
    cv2.MORPH_OPEN,
    kernel
)

# Find contours
contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Number of contours found:", len(contours))

# Copy image for drawing
result = image.copy()

# Minimum area
minimum_area = 1000

# Process contours
for contour in contours:

    # Calculate contour area
    area = cv2.contourArea(contour)

    if area < minimum_area:
        continue

    print("\n--- Blue Object ---")

    # Bounding box
    x, y, width, height = cv2.boundingRect(contour)

    print("Width:", width, "pixels")
    print("Height:", height, "pixels")
    print("Area:", area, "pixels^2")

    # Equivalent diameter
    equivalent_diameter = 2 * math.sqrt(area / math.pi)

    print(
        "Equivalent diameter:",
        round(equivalent_diameter, 2),
        "pixels"
    )

    # Perimeter
    perimeter = cv2.arcLength(
        contour,
        True
    )

    print(
        "Perimeter:",
        round(perimeter, 2),
        "pixels"
    )

    # Circularity
    if perimeter != 0:
        circularity = (
            4 * math.pi * area
        ) / (perimeter * perimeter)
    else:
        circularity = 0

    print(
        "Circularity:",
        round(circularity, 3)
    )

    # Shape approximation
    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    number_of_vertices = len(approximation)

    if circularity > 0.75:
        shape = "ROUND / CIRCLE"
    elif number_of_vertices == 3:
        shape = "TRIANGLE"
    elif number_of_vertices == 4:
        shape = "RECTANGLE / SQUARE"
    else:
        shape = "IRREGULAR"

    print("Shape:", shape)

    # Centroid
    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        centroid_x = int(
            moments["m10"] / moments["m00"]
        )

        centroid_y = int(
            moments["m01"] / moments["m00"]
        )

        print(
            "Centroid:",
            (centroid_x, centroid_y),
            "pixels"
        )

        # Draw centroid
        cv2.circle(
            result,
            (centroid_x, centroid_y),
            6,
            (0, 0, 255),
            -1
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

    # Display information
    cv2.putText(
        result,
        f"Shape: {shape}",
        (x, y - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Size: {width} x {height} px",
        (x, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

# Show result
cv2.imshow(
    "Blue Object Analysis",
    result
)

# Show mask
cv2.imshow(
    "Cleaned Blue Mask",
    clean_mask
)

# Wait for key
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()