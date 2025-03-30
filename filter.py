import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
time.sleep(2)  # Allow camera to warm up

# Initialize MediaPipe Hands & Selfie Segmentation
mpHands = mp.solutions.hands  # Load hand tracking module
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Set detection thresholds
mpDraw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks
mpSelfieSegmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)  # Load background segmentation model

# Define color filters (RGB format)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128)]
selected_color = (0, 0, 0)  # Default background color (black)

# Frame counter for debounce mechanism
color_selection_frame_count = 0  # Counter to ensure stable selection
selected_color_index = -1  # Index of selected color

# Function to apply background filter
def apply_filter(img, mask, color):
    color_layer = np.full_like(img, color, dtype=np.uint8)  # Create a solid color layer
    blurred_img = cv2.GaussianBlur(img, (15, 15), 10)  # Apply Gaussian blur to the original image
    # Blend the original image with the color filter where the mask is True
    filtered_img = np.where(mask[:, :, None], cv2.addWeighted(img, 0.6, color_layer, 0.4, 0), blurred_img)
    return filtered_img

while True:
    success, img = cap.read()  # Read frame from webcam
    if not success:
        continue  # Skip if frame is not captured
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for MediaPipe processing
    results = hands.process(imgRGB)  # Process hand tracking
    seg_results = mpSelfieSegmentation.process(imgRGB)  # Process background segmentation

    mask = seg_results.segmentation_mask > 0.5  # Generate binary mask for background segmentation
    img = apply_filter(img, mask, selected_color)  # Apply selected background filter

    # Draw color selection boxes
    for i, color in enumerate(colors):
        x, y = 50 + i * 60, 30  # Position color boxes in a row
        cv2.rectangle(img, (x, y), (x + 50, y + 50), color, -1)  # Draw color box
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)  # Draw hand landmarks

            # Detect index finger tip position
            index_finger_x = int(hand_landmarks.landmark[8].x * img.shape[1])
            index_finger_y = int(hand_landmarks.landmark[8].y * img.shape[0])

            # Detect if the finger is over a color box
            for i, color in enumerate(colors):
                x, y = 50 + i * 60, 30  # Get box position
                if x < index_finger_x < x + 50 and y < index_finger_y < y + 50:
                    if selected_color_index == i:
                        color_selection_frame_count += 1  # Increase frame count if finger stays
                    else:
                        selected_color_index = i
                        color_selection_frame_count = 1  # Reset counter when switching colors

                    # Confirm selection after finger stays for 10 frames
                    if color_selection_frame_count > 10:
                        selected_color = color  # Update selected color
                        color_selection_frame_count = 0  # Reset counter
                    break  # Stop checking once a color is selected

    cv2.imshow("Hand Detection with Color Filters", img)  # Display the processed frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop when 'q' is pressed

cap.release()
cv2.destroyAllWindows()  # Release resources and close windows
