import cv2

# initialize the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the camera
video_capture = cv2.VideoCapture(0)  # 0 represents the default camera, change to another number if you have multiple cameras.

while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()

    # detect humans in the current frame
    (humans, _) = hog.detectMultiScale(frame, winStride=(10, 10),
                                       padding=(32, 32), scale=1.1)

    # getting no. of human detected
    print('Human Detected : ', len(humans))

    # loop over all detected humans and draw rectangles
    for (x, y, w, h) in humans:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

    # display the output frame
    cv2.imshow("Camera", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
