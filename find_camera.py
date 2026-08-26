import cv2

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open Camera 1.")
    exit()

print("Camera 1 opened successfully!")
print("Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    cv2.imshow("Camera 1 Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()