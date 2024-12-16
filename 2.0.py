import cv2
import mediapipe as mp
import pyautogui
import time  # Added to handle gesture timing

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Gesture thresholds
SWIPE_THRESHOLD = 30  # Reduced distance for swipe
PINCH_THRESHOLD = 30  # Distance for pinch

# Gesture tracking variables
prev_position = None
gesture_detected = None
prev_time = 0  # Added for gesture timing

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror-like effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmark coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw points for debugging
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

                # Detect pinch (Zoom In/Out) based on distance
                pinch_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

                if pinch_distance < PINCH_THRESHOLD:
                    gesture_detected = "Zoom In"
                    pyautogui.hotkey('ctrl', '+')
                elif pinch_distance > PINCH_THRESHOLD + 20:
                    gesture_detected = "Zoom Out"
                    pyautogui.hotkey('ctrl', '-')

                # Swipe detection for next/previous slide
                current_time = time.time()  # Current time for timing gestures
                if prev_position:
                    delta_x = index_x - prev_position[0]  # Horizontal movement
                    delta_y = index_y - prev_position[1]  # Vertical movement

                    # Detect swipe up (next slide) and swipe down (previous slide)
                    if abs(delta_y) > SWIPE_THRESHOLD and (current_time - prev_time > 0.5):  # Add timing constraint
                        prev_time = current_time  # Update the time
                        if delta_y < 0:
                            gesture_detected = "Next Slide"
                            pyautogui.press("down")
                        elif delta_y > 0:
                            gesture_detected = "Previous Slide"
                            pyautogui.press("up")

                prev_position = (index_x, index_y)

        # Display detected gesture
        if gesture_detected:
            cv2.putText(frame, f"Gesture: {gesture_detected}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            gesture_detected = None

        # Show the frame
        cv2.imshow("Gesture Control", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
