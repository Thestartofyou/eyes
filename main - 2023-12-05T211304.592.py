import dlib
import keyboard

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You'll need to download this file

# Initialize variables for blink detection
blink_counter = 0

# Function to check if eyes are closed
def are_eyes_closed(shape):
    left_eye_ratio = (shape[42].y - shape[38].y + shape[41].y - shape[37].y) / (2.0 * (shape[40].x - shape[36].x))
    right_eye_ratio = (shape[47].y - shape[43].y + shape[46].y - shape[44].y) / (2.0 * (shape[45].x - shape[42].x))
    
    avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
    return avg_eye_ratio < 0.2  # Adjust this threshold based on your conditions

# Function to turn off lights (replace with your actual code)
def turn_off_lights():
    print("Turning off lights")
    # Replace this with code to control your lights, e.g., using a smart home API

# Capture video from your webcam (you may need to adjust the index)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces in the frame
    faces = detector(frame)

    for face in faces:
        shape = predictor(frame, face)

        # Check if eyes are closed
        if are_eyes_closed(shape.parts()):
            blink_counter += 1
        else:
            blink_counter = 0

        # If blink counter reaches 2, turn off lights
        if blink_counter == 2:
            turn_off_lights()
            blink_counter = 0  # Reset counter

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

