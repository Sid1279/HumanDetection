import cv2

# Load the jersey image with alpha channel (RGBA format)
jersey_image = cv2.imread('jersey.png', cv2.IMREAD_UNCHANGED)

# Check if the image is loaded correctly
if jersey_image is None:
    print("Error: Failed to load the jersey image.")
else:
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

        # loop over all detected humans and overlay the jersey
        for (x, y, w, h) in humans:
            center_x = x + w // 2
            center_y = y + h // 2 - h // 6
            
            # Resize the jersey to match the width of the bounding box
            jersey_width = int(0.8*w)
            jersey_height = int(jersey_image.shape[0] * 0.8 * jersey_width / jersey_image.shape[1])
            jersey_resized = cv2.resize(jersey_image, (jersey_width, jersey_height))

            # Calculate the top-left corner coordinates of the jersey overlay
            jersey_x = center_x - jersey_width // 2
            jersey_y = center_y - jersey_height // 2 + jersey_height // 6

            # Make sure the jersey is entirely inside the frame
            jersey_x = max(jersey_x, 0)
            jersey_y = max(jersey_y, 0)

            # Calculate the overlapping region of the jersey and frame
            overlapping_x = max(0, -jersey_x)
            overlapping_y = max(0, -jersey_y)

            # Overlay the jersey on the frame using the alpha channel
            for c in range(3):
                frame_slice = frame[jersey_y + overlapping_y:jersey_y + overlapping_y + jersey_height, jersey_x + overlapping_x:jersey_x + overlapping_x + jersey_width, c]
                jersey_slice = jersey_resized[overlapping_y:overlapping_y + frame_slice.shape[0], overlapping_x:overlapping_x + frame_slice.shape[1], c]
                alpha_slice = jersey_resized[overlapping_y:overlapping_y + frame_slice.shape[0], overlapping_x:overlapping_x + frame_slice.shape[1], 3] / 255.0

                frame_slice[:, :] = frame_slice * (1 - alpha_slice) + jersey_slice * alpha_slice

        # display the output frame
        cv2.imshow("Camera", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
