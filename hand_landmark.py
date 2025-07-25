# Import necessary libraries
import cv2  # OpenCV for image and video processing
import mediapipe as mp  # MediaPipe for machine learning models (face mesh, etc.)

# Initialize MediaPipe drawing utilities
mp_drawing= mp.solutions.drawing_utils
# Define drawing specifications (circle radius and line thickness)
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green mesh

# Initialize the MediaPipe hand mesh model
mp_hands = mp.solutions.hands

# Start capturing video from the default webcam (0 refers to the first webcam)
video = cv2.VideoCapture(0)

# Load the FaceMesh model with confidence thresholds
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    # Infinite loop to process video frame-by-frame
    while True:
        # Read a single frame from the webcam
        ret, frame = video.read()
        if not ret:
            break  # If no frame is captured, break the loop

        # Convert the frame from BGR (OpenCV default) to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Improve performance by marking the image as not writeable (no unnecessary copying)
        frame_rgb.flags.writeable = False

        # Process the RGB frame to detect face mesh landmarks
        hand_results = hands.process(frame_rgb)

        # Set the image back to writeable for further OpenCV processing (like drawing)
        frame_rgb.flags.writeable = True

        # Convert the frame back from RGB to BGR for OpenCV display
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
             for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )


        # Display the processed frame in a window titled "Face Mesh"
        cv2.imshow("Face Mesh", frame)

        # Wait for 1ms; if 'q' key is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam resource
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
