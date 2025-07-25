# Import necessary libraries
import cv2  # OpenCV for image and video processing
import mediapipe as mp  # MediaPipe for machine learning models (face mesh, etc.)

# Initialize MediaPipe drawing utilities
mp_drawings = mp.solutions.drawing_utils

# Initialize the MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh

# Define drawing specifications (circle radius and line thickness)
drawing_spec = mp_drawings.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green mesh


# Start capturing video from the default webcam (0 refers to the first webcam)
video = cv2.VideoCapture(0)

# Load the FaceMesh model with confidence thresholds
with mp_face_mesh.FaceMesh(
    max_num_faces=1,                     # Detect only 1 face (you can increase this if needed)
    min_detection_confidence=0.5,       # Minimum confidence for face detection
    min_tracking_confidence=0.5         # Minimum confidence for landmark tracking
) as face_mesh:

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
        results = face_mesh.process(frame_rgb)

        # Set the image back to writeable for further OpenCV processing (like drawing)
        frame_rgb.flags.writeable = True

        # Convert the frame back from RGB to BGR for OpenCV display
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # If landmarks are detected on a face
        if results.multi_face_landmarks:
            # Loop through all the faces detected (just 1 in this case)
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh landmarks and connections on the frame
                mp_drawings.draw_landmarks(
                    image=frame,  # Frame to draw on
                    landmark_list=face_landmarks,  # Detected face landmarks
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # Predefined mesh connections
                    landmark_drawing_spec=drawing_spec,  # Landmark point style
                    connection_drawing_spec=drawing_spec  # Line/connection style
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
