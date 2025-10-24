# detect_drone.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
from huggingface_hub import hf_hub_download


class DroneDetector:
    def __init__(self, camera_id=0, model_path='yolov8n.pt'):
        """Initialize drone detector with camera and YOLO model"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load YOLO model
        model_path = "models/unidrone_yolov8m_640px/weights/best.pt"
        self.model = YOLO(model_path)

        # Target classes (14 = 'bird' in COCO, good drone proxy)
        self.target_classes = [0]    # Add more classes as needed
        self.confidence_threshold = 0.1
        
        # For calculating turret commands
        self.frame_center_x = 320  # Half of frame width
        self.frame_center_y = 240  # Half of frame height

    def detect_targets(self, frame):
        """Run YOLO detection on frame"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in self.target_classes and confidence > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Calculate pixel error from center
                        error_x, error_y = self.calculate_turret_error((center_x, center_y))
                        
                        # Convert to motor commands
                        pan_degrees, tilt_degrees = self.calculate_motor_commands(error_x, error_y)
                        
                        # Print movement commands
                        print(f"Target at ({center_x}, {center_y})")
                        print(f"Pixel Error: ({error_x}, {error_y})")
                        print(f"Motor Command: Pan {pan_degrees:.2f}°, Tilt {tilt_degrees:.2f}°")
                        print(f"---")
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (center_x, center_y),
                            'confidence': confidence,
                            'class_id': class_id,
                            'motor_commands': (pan_degrees, tilt_degrees)  # Add this
                        })
        
        return detections
    # In your detect_drone.py, add this method:
    def calculate_motor_commands(self, error_x, error_y):
        """Convert pixel errors to motor movement commands"""
        
        # Camera field of view (you need to measure/calibrate this)
        CAMERA_FOV_HORIZONTAL = 62.0  # degrees (typical webcam)
        CAMERA_FOV_VERTICAL = 48.0    # degrees
        
        # Frame dimensions
        frame_width = 640
        frame_height = 480
        
        # Convert pixels to degrees
        degrees_per_pixel_x = CAMERA_FOV_HORIZONTAL / frame_width
        degrees_per_pixel_y = CAMERA_FOV_VERTICAL / frame_height
        
        # Calculate required movement in degrees
        pan_movement = error_x * degrees_per_pixel_x    # positive = move right
        tilt_movement = -error_y * degrees_per_pixel_y  # negative because Y is inverted
        
        return pan_movement, tilt_movement
    def calculate_turret_error(self, target_center):
        """Calculate how much turret needs to move to center target"""
        center_x, center_y = target_center
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        return error_x, error_y
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and info on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw confidence
            label = f'Drone: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display turret error
            error_x, error_y = self.calculate_turret_error((center_x, center_y))
            error_text = f'Error: ({error_x}, {error_y})'
            cv2.putText(frame, error_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Draw frame center crosshair
        cv2.line(frame, (self.frame_center_x-20, self.frame_center_y), 
                (self.frame_center_x+20, self.frame_center_y), (255, 255, 255), 1)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y-20), 
                (self.frame_center_x, self.frame_center_y+20), (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("Starting drone detection... Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect targets
            detections = self.detect_targets(frame)
            
            # Draw results
            frame = self.draw_detections(frame, detections)
            
            # Display info
            fps_text = f'Detections: {len(detections)}'
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('CastleWatch - Drone Detection', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DroneDetector()
    detector.run()