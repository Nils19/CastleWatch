import cv2
import serial
import time

# PID controller function
def pid_output(error, previous_error, integral, Kp, Ki, Kd, dt):
    integral += error * dt
    derivative = (error - previous_error) / dt
    command = Kp * error + Ki * integral + Kd * derivative
    return command, integral, error

def compute_target_coordinates(cap, face_cascade):
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    # Optional: resize frame to speed up detection
    frame = cv2.resize(frame, (320, 240))

    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)

    if len(faces) == 0:
        cv2.imshow("Face Tracking", frame)
        return None, frame_center, frame

    x, y, w, h = faces[0]
    face_center = (x + w // 2, y + h // 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, face_center, 5, (0, 0, 255), -1)
    cv2.line(frame, face_center, frame_center, (0, 255, 255), 2)
    cv2.imshow("Face Tracking", frame)

    return face_center, frame_center, frame

def compute_aim_error(face_center, frame_center):
    if face_center is None:
        return 0, 0  
    error_x = frame_center[0] - face_center[0]
    error_y = face_center[1] - frame_center[1]
    print(f"Error: {error_x}, {error_y}")
    return error_x, error_y

# Arduino setup
arduino_port = "COM8"  # adjust to your port
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

# OpenCV setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Give camera time to initialize
print("Initializing camera...")
for _ in range(10):
    ret, _ = cap.read()
    time.sleep(0.1)
print("Camera ready!")

# PID parameters
Kp, Ki, Kd = 0.06, 0.001, 0
dt = 0.1
integral_x = integral_y = 0
previous_error_x = previous_error_y = 0

# Main loop
try:
    while True:
        face_center, frame_center, frame = compute_target_coordinates(cap, face_cascade)
        if frame is None:
            print("Error reading frame")
            break

        error_x, error_y = compute_aim_error(face_center, frame_center)
        command_x, integral_x, previous_error_x = pid_output(error_x, previous_error_x, integral_x, Kp, Ki, Kd, dt)
        command_y, integral_y, previous_error_y = pid_output(error_y, previous_error_y, integral_y, Kp, Ki, Kd, dt)

        # Send commands to Arduino
        ser.write(f"{command_x:.2f},{command_y:.2f}\n".encode())

        # Refresh OpenCV window and allow exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Prevent serial buffer overflow
        while ser.in_waiting:
            ser.readline()

finally:
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("Cleanup complete")
