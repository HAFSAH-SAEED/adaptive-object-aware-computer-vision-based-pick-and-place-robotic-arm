
import cv2

# Open the default camera
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("Camera opened successfully!")
print("Press Q to quit.")

while True:

    # Capture one frame from the camera
    success, frame = camera.read()

    # Check if the frame was captured
    if not success:
        print("Could not read frame.")
        break

    # Display the live camera frame
    cv2.imshow("Live Camera", frame)

    # Press Q to stop the camera
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()