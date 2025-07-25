import cv2  
import mediapipe as mp 
import time
from directkeys import right_pressed, left_pressed, PressKey, ReleaseKey

break_key_pressed=left_pressed # Key code for brake
accelerato_key_pressed=right_pressed # Key code for accelerator
time.sleep(2.0)  # Allow time to switch to the game window
current_key_pressed = set() # Set to keep track of currently pressed keys

mp_drawing= mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1) 

mp_hands = mp.solutions.hands

tip_ids = [4, 8, 12, 16, 20]  # List of tip landmark IDs for fingers

video = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        keyPressed = False # Variable to track if a key is pressed
        break_pressed=False # Variable to track if the brake key is pressed
        accelerator_pressed=False # Variable to track if the accelerator key is pressed
        key_count=0 # Counter for the number of keys pressed
        key_pressed=0 # Variable to store the currently pressed key
        ret, frame = video.read() 
        if not ret:
            break 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        hand_results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        lmlist = []
        if hand_results.multi_hand_landmarks:
             for hand_landmarks in hand_results.multi_hand_landmarks:
                    myHands=hand_results.multi_hand_landmarks[0] # Get the first detected hand landmarks
                    for id,lm in enumerate(myHands.landmark): # Iterate through each landmark
                        h,w,c=frame.shape # Get the height, width, and number of channels
                        cx, cy = int(lm.x * w), int(lm.y * h) # Convert normalized coordinates to pixel values
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)# Draw a circle at each landmark position
                        # print(f"Landmark {id}: ({cx}, {cy})") # Print the landmark ID and its coordinates
                        lmlist.append([id, cx, cy])
        
                    mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        fingers= [] # List to store the state of each finger (open or closed)
        if len(lmlist) != 0:
             if lmlist[tip_ids[0]][1] < lmlist[tip_ids[0]-1][1]: # Check if the thumb is open
                fingers.append(1) # Thumb is open
             else:
                fingers.append(0)
             for id in range(1,5): # Iterate through the other fingers
                if lmlist[tip_ids[id]][2] < lmlist[tip_ids[id]-2][2]: # Check if the finger is open
                    fingers.append(1)
                else:
                    fingers.append(0)
             total= fingers.count(1)
             if total == 0:
                cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED) # Draw a rectangle for no fingers open
                cv2.putText(frame, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 5) # Display "BRAKE"
                PressKey(break_key_pressed) # Press the brake key
                break_pressed=True # Set brake pressed to True
                current_key_pressed.add(break_key_pressed) # Add brake key to the current pressed keys
                key_pressed=break_key_pressed # Set the currently pressed key to brake
                keyPressed = True # Set keyPressed to True
                key_count=key_count+1 # Increment the key count
             elif total == 5:
                cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, " GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 5)
                PressKey(accelerato_key_pressed)
                key_pressed=accelerato_key_pressed
                accelerator_pressed=True
                keyPressed = True
                current_key_pressed.add(accelerato_key_pressed)
                key_count=key_count+1
        if not keyPressed and len(current_key_pressed) != 0: # If no key is pressed and there are currently pressed keys
            for key in current_key_pressed: # Release all currently pressed keys
                ReleaseKey(key)
            current_key_pressed = set()
        elif key_count==1 and len(current_key_pressed)==2:  # If one key is pressed and two keys are currently pressed   
            for key in current_key_pressed:             
                if key_pressed!=key:
                    ReleaseKey(key)
            current_key_pressed = set()
            for key in current_key_pressed:
                ReleaseKey(key)
            current_key_pressed = set()
             
            #  if lmlist[8][2] < lmlist[6][2]:
            #       print("Open")
            #  else:
            #       print("Closed")
            
        cv2.imshow("Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()