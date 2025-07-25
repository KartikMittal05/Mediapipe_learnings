import cv2
import mediapipe as mp

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Face Mesh and Hands model setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Start webcam
video = cv2.VideoCapture(0)

# Load FaceMesh and Hands models
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh, mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = video.read()
        frame = cv2.resize(frame, (450, 450)) 
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Process face and hand landmarks
        face_results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)

        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # Show frame
        cv2.imshow("Face & Hand Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
