from __future__ import annotations

import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from collections import deque
import math


class PredictiveDroneTurret:
    def __init__(self, arduino_port='/dev/cu.usbmodem1401', camera_id=0):
        # Arduino connection
        self.arduino_port = arduino_port
        self.baud_rate = 9600
        self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
        time.sleep(2)
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print('Error: Could not open webcam.')
            exit()
        
        # YOLO drone detection model
        model_path = "models/unidrone_yolov8m_640px/weights/best.pt"
        self.model = YOLO(model_path)
        self.target_classes = [0]
        self.confidence_threshold = 0.3
        
        # Frame parameters
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        # Enhanced PID parameters with different gains for tracking vs holding
        self.tracking_gains = {'Kp': 0.08, 'Ki': 0.001, 'Kd': 0.03}
        self.holding_gains = {'Kp': 0.04, 'Ki': 0.0005, 'Kd': 0.01}
        self.current_gains = self.tracking_gains
        
        self.dt = 0.05  # Faster update rate for better tracking
        self.integral_x = self.integral_y = 0
        self.previous_error_x = self.previous_error_y = 0
        
        # PREDICTIVE TRACKING SYSTEM
        self.target_history = deque(maxlen=10)  # Store last 10 positions with timestamps
        self.velocity_history = deque(maxlen=5)  # Store velocity estimates
        self.prediction_time = 0.3  # Predict 300ms ahead (adjustable)
        self.min_velocity_threshold = 5.0  # Minimum pixels/sec to enable prediction
        
        # KALMAN FILTER for smoother tracking
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 states (x,y,vx,vy), 2 measurements (x,y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kalman_initialized = False
        
        # MULTI-TARGET TRACKING
        self.tracked_targets = {}  # Store multiple target trackers
        self.max_targets = 3
        self.target_timeout = 1.0  # Remove targets not seen for 1 second
        
        # ENGAGEMENT ZONES
        self.engagement_zone = {'x_min': 100, 'x_max': 540, 'y_min': 80, 'y_max': 400}
        self.priority_zone = {'x_min': 200, 'x_max': 440, 'y_min': 160, 'y_max': 320}
        
        # PERFORMANCE METRICS
        self.tracking_stats = {
            'shots_on_target': 0,
            'total_tracking_time': 0,
            'average_error': deque(maxlen=100)
        }
        
        print('Advanced Predictive Drone Turret initialized!')
        self.initialize_camera()

    def initialize_camera(self):
        """Enhanced camera initialization with auto-exposure control"""
        print('Initializing camera...')
        
        # Try to set manual exposure for consistent detection
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Fast exposure for moving targets
        except:
            print("Could not set manual exposure")
        
        for _ in range(15):  # More frames for better initialization
            ret, _ = self.cap.read()
            time.sleep(0.05)
        print('Camera ready!')

    def detect_drones(self, frame):
        """Enhanced drone detection with confidence boosting"""
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
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Boost confidence for targets in priority zones
                        zone_boost = 0
                        if self.is_in_priority_zone(center_x, center_y):
                            zone_boost = 0.1
                        elif self.is_in_engagement_zone(center_x, center_y):
                            zone_boost = 0.05
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (center_x, center_y),
                            'confidence': confidence + zone_boost,
                            'class_id': class_id,
                            'area': area,
                            'timestamp': time.time()
                        })
        
        return detections

    def is_in_engagement_zone(self, x, y):
        """Check if target is in engagement zone"""
        return (self.engagement_zone['x_min'] <= x <= self.engagement_zone['x_max'] and
                self.engagement_zone['y_min'] <= y <= self.engagement_zone['y_max'])

    def is_in_priority_zone(self, x, y):
        """Check if target is in priority zone (center of frame)"""
        return (self.priority_zone['x_min'] <= x <= self.priority_zone['x_max'] and
                self.priority_zone['y_min'] <= y <= self.priority_zone['y_max'])

    def update_target_tracking(self, detections):
        """Update Kalman filter and velocity estimation"""
        current_time = time.time()
        
        if not detections:
            return None
        
        # Get best target (highest confidence in engagement zone)
        valid_targets = [d for d in detections if self.is_in_engagement_zone(d['center'][0], d['center'][1])]
        if not valid_targets:
            valid_targets = detections
        
        best_target = max(valid_targets, key=lambda x: x['confidence'])
        target_pos = best_target['center']
        
        # Initialize Kalman filter if needed
        if not self.kalman_initialized:
            self.kalman.statePre = np.array([target_pos[0], target_pos[1], 0, 0], dtype=np.float32)
            self.kalman.statePost = np.array([target_pos[0], target_pos[1], 0, 0], dtype=np.float32)
            self.kalman_initialized = True
            
        # Predict and update Kalman filter
        prediction = self.kalman.predict()
        measurement = np.array([[target_pos[0]], [target_pos[1]]], dtype=np.float32)
        self.kalman.correct(measurement)
        
        # Store position history for velocity calculation
        self.target_history.append({
            'pos': target_pos,
            'time': current_time,
            'filtered_pos': (prediction[0], prediction[1])
        })
        
        # Calculate velocity if we have enough history
        if len(self.target_history) >= 3:
            self.calculate_velocity()
        
        return best_target

    def calculate_velocity(self):
        """Calculate target velocity using recent position history"""
        if len(self.target_history) < 3:
            return
        
        # Use last 3 points for velocity calculation
        recent_points = list(self.target_history)[-3:]
        
        # Calculate velocity using linear regression for smoothness
        times = [p['time'] for p in recent_points]
        x_positions = [p['pos'][0] for p in recent_points]
        y_positions = [p['pos'][1] for p in recent_points]
        
        if len(times) >= 2:
            dt = times[-1] - times[0]
            if dt > 0:
                vx = (x_positions[-1] - x_positions[0]) / dt
                vy = (y_positions[-1] - y_positions[0]) / dt
                
                # Store velocity
                self.velocity_history.append({
                    'vx': vx,
                    'vy': vy,
                    'speed': math.sqrt(vx*vx + vy*vy),
                    'time': times[-1]
                })

    def predict_target_position(self, current_pos):
        """PREDICTIVE TRACKING: Calculate where target will be"""
        if len(self.velocity_history) < 2:
            return current_pos  # No prediction without velocity data
        
        # Get latest velocity
        latest_velocity = self.velocity_history[-1]
        
        # Only predict if target is moving fast enough
        if latest_velocity['speed'] < self.min_velocity_threshold:
            return current_pos
        
        # Predict future position
        predicted_x = current_pos[0] + (latest_velocity['vx'] * self.prediction_time)
        predicted_y = current_pos[1] + (latest_velocity['vy'] * self.prediction_time)
        
        # Constrain to frame bounds
        predicted_x = max(0, min(self.frame_width, predicted_x))
        predicted_y = max(0, min(self.frame_height, predicted_y))
        
        return (int(predicted_x), int(predicted_y))

    def adaptive_pid_gains(self, target_speed, error_magnitude):
        """Adjust PID gains based on target behavior"""
        if target_speed > 50:  # Fast moving target
            return {'Kp': 0.12, 'Ki': 0.002, 'Kd': 0.04}
        elif error_magnitude < 20:  # Close to target
            return {'Kp': 0.03, 'Ki': 0.0008, 'Kd': 0.015}
        else:  # Default tracking
            return self.tracking_gains

    def compute_target_coordinates(self):
        """Enhanced target computation with prediction"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None

        # Detect drones
        detections = self.detect_drones(frame)
        
        # Update tracking
        best_target = self.update_target_tracking(detections)
        
        current_pos = None
        predicted_pos = None
        
        if best_target:
            current_pos = best_target['center']
            predicted_pos = self.predict_target_position(current_pos)
        
        frame_center = (self.frame_center_x, self.frame_center_y)
        
        # Draw enhanced visualizations
        frame = self.draw_enhanced_detections(frame, detections, current_pos, predicted_pos, frame_center)
        
        return current_pos, predicted_pos, frame_center, frame

    def draw_enhanced_detections(self, frame, detections, current_pos, predicted_pos, frame_center):
        """Enhanced visualization with prediction, zones, and stats"""
        
        # Draw engagement zones
        cv2.rectangle(frame, 
                     (self.engagement_zone['x_min'], self.engagement_zone['y_min']),
                     (self.engagement_zone['x_max'], self.engagement_zone['y_max']),
                     (100, 100, 100), 1)
        
        cv2.rectangle(frame, 
                     (self.priority_zone['x_min'], self.priority_zone['y_min']),
                     (self.priority_zone['x_max'], self.priority_zone['y_max']),
                     (150, 150, 150), 1)
        
        # Draw all detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            
            # Color coding based on zones
            if self.is_in_priority_zone(center_x, center_y):
                color = (0, 255, 0)  # Green for priority
            elif self.is_in_engagement_zone(center_x, center_y):
                color = (0, 255, 255)  # Yellow for engagement
            else:
                color = (0, 150, 255)  # Orange for out of zone
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            label = f'D: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw frame center
        cv2.line(frame, (frame_center[0]-15, frame_center[1]), 
                (frame_center[0]+15, frame_center[1]), (255, 0, 0), 2)
        cv2.line(frame, (frame_center[0], frame_center[1]-15), 
                (frame_center[0], frame_center[1]+15), (255, 0, 0), 2)
        
        # Draw current target and prediction
        if current_pos:
            # Current position (solid circle)
            cv2.circle(frame, current_pos, 8, (0, 255, 0), 2)
            cv2.putText(frame, "CURRENT", (current_pos[0]-30, current_pos[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            if predicted_pos and predicted_pos != current_pos:
                # Predicted position (dashed circle)
                cv2.circle(frame, predicted_pos, 8, (255, 255, 0), 2)
                cv2.putText(frame, "LEAD", (predicted_pos[0]-20, predicted_pos[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw prediction vector
                cv2.arrowedLine(frame, current_pos, predicted_pos, (255, 255, 0), 2)
                
                # Draw targeting line to predicted position
                cv2.line(frame, predicted_pos, frame_center, (0, 255, 255), 2)
            else:
                # Draw targeting line to current position
                cv2.line(frame, current_pos, frame_center, (0, 255, 255), 2)
        
        # Draw velocity info
        if self.velocity_history:
            latest_vel = self.velocity_history[-1]
            speed_text = f'Speed: {latest_vel["speed"]:.1f} px/s'
            cv2.putText(frame, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            vel_text = f'Vel: ({latest_vel["vx"]:.1f}, {latest_vel["vy"]:.1f})'
            cv2.putText(frame, vel_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw stats
        detection_text = f'Targets: {len(detections)}'
        cv2.putText(frame, detection_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        prediction_text = f'Prediction: {self.prediction_time*1000:.0f}ms'
        cv2.putText(frame, prediction_text, (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('CastleWatch - Predictive Drone Turret', frame)
        return frame

    def compute_aim_error(self, target_pos, frame_center):
        """Calculate aiming error using predicted position if available"""
        if target_pos is None:
            return 0, 0
        
        error_x = frame_center[0] - target_pos[0]
        error_y = target_pos[1] - frame_center[1]  # Inverted Y
        return error_x, error_y

    def pid_output(self, error, previous_error, integral, gains, dt):
        """Enhanced PID with adaptive gains"""
        Kp, Ki, Kd = gains['Kp'], gains['Ki'], gains['Kd']
        
        integral += error * dt
        # Prevent integral windup
        integral = max(-50, min(50, integral))
        
        derivative = (error - previous_error) / dt
        command = Kp * error + Ki * integral + Kd * derivative
        
        return command, integral, error

    def send_turret_command(self, command_x, command_y):
        """Send movement command with rate limiting"""
        command_str = f"{command_x:.2f},{command_y:.2f}\n"
        self.ser.write(command_str.encode())
        
        # Read Arduino messages
        while self.ser.in_waiting > 0:
            try:
                arduino_msg = self.ser.readline().decode('utf-8').strip()
                if arduino_msg and not arduino_msg.startswith("Pos:"):
                    print(f"Arduino: {arduino_msg}")
            except:
                pass

    def run(self):
        """Enhanced main loop with predictive tracking"""
        print("Starting PREDICTIVE drone tracking turret...")
        print("Controls:")
        print("  'q' - quit")
        print("  'r' - reset PID")
        print("  '+' - increase prediction time")
        print("  '-' - decrease prediction time")
        print("  'f' - toggle fast/slow mode")
        
        fast_mode = False
        
        try:
            while True:
                start_time = time.time()
                
                # Get detection and prediction results
                current_pos, predicted_pos, frame_center, frame = self.compute_target_coordinates()

                if frame is None:
                    print('Error reading frame')
                    break

                # Use predicted position for aiming if available, otherwise current
                target_pos = predicted_pos if predicted_pos else current_pos
                error_x, error_y = self.compute_aim_error(target_pos, frame_center)
                
                # Adaptive PID gains
                target_speed = self.velocity_history[-1]['speed'] if self.velocity_history else 0
                error_magnitude = math.sqrt(error_x*error_x + error_y*error_y)
                adaptive_gains = self.adaptive_pid_gains(target_speed, error_magnitude)
                
                # Calculate PID commands
                command_x, self.integral_x, self.previous_error_x = self.pid_output(
                    error_x, self.previous_error_x, self.integral_x, adaptive_gains, self.dt
                )
                command_y, self.integral_y, self.previous_error_y = self.pid_output(
                    error_y, self.previous_error_y, self.integral_y, adaptive_gains, self.dt
                )

                # Send commands
                self.send_turret_command(command_x, command_y)
                
                # Update stats
                if target_pos:
                    self.tracking_stats['average_error'].append(error_magnitude)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.integral_x = self.integral_y = 0
                    print("PID reset")
                elif key == ord('+'):
                    self.prediction_time = min(1.0, self.prediction_time + 0.05)
                    print(f"Prediction time: {self.prediction_time:.2f}s")
                elif key == ord('-'):
                    self.prediction_time = max(0.1, self.prediction_time - 0.05)
                    print(f"Prediction time: {self.prediction_time:.2f}s")
                elif key == ord('f'):
                    fast_mode = not fast_mode
                    self.dt = 0.03 if fast_mode else 0.05
                    print(f"Fast mode: {fast_mode}")

                # Adaptive frame rate
                loop_time = time.time() - start_time
                sleep_time = max(0, self.dt - loop_time)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print('Interrupted by user')
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up and print final stats"""
        print("\n=== TRACKING STATISTICS ===")
        if self.tracking_stats['average_error']:
            avg_error = sum(self.tracking_stats['average_error']) / len(self.tracking_stats['average_error'])
            print(f"Average tracking error: {avg_error:.1f} pixels")
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.ser.close()
        print('Advanced turret system shutdown complete')


if __name__ == "__main__":
    turret = PredictiveDroneTurret(
        arduino_port='/dev/cu.usbmodem1401',
        camera_id=0
    )
    turret.run()