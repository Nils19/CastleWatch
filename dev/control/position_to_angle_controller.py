from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import math  

import serial


class PositionToAngleController:
    def __init__(
        self,
        arduino_port='COM4',
        baud_rate=9600,
        log_file_path='C:\\Users\\gemv\\repos\\private\\CastleWatch\\out\\detection.log',
        frame_width=640,
        frame_height=480,
        camera_fov_horizontal=62.0,  # degrees
        camera_fov_vertical=48.0,  # degrees
        kp_pan=0.5,  # Proportional gain for pan (tune this)
        kp_tilt=0.5,  # Proportional gain for tilt (tune this)
    ):
        """Initialize the controller with Arduino and log file settings"""
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.log_file_path = Path(log_file_path)
        self.ser = None
        self.running = False
        self.thread = None

        # --- Store frame dimensions ---
        self.frame_width = frame_width
        self.frame_height = frame_height

        # --- Calculate and store focal lengths in pixels ---
        # f_x = (width / 2) / tan(HFOV / 2)
        self.focal_length_x = (self.frame_width / 2) / math.tan(
            math.radians(camera_fov_horizontal / 2)
        )
        # f_y = (height / 2) / tan(VFOV / 2)
        self.focal_length_y = (self.frame_height / 2) / math.tan(
            math.radians(camera_fov_vertical / 2)
        )

        print(f"Calculated focal lengths: fx={self.focal_length_x:.2f}px, fy={self.focal_length_y:.2f}px")

        # --- Store PID gains ---
        self.Kp_pan = kp_pan
        self.Kp_tilt = kp_tilt

    def start(self):
        """Start monitoring the log file and sending commands to Arduino"""
        if self.running:
            print('Controller is already running')
            return

        try:
            # Initialize Arduino connection
            self.ser = serial.Serial(
                self.arduino_port, self.baud_rate, timeout=1,
            )
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Connected to Arduino on {self.arduino_port}")

            # Start monitoring thread
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print('Position to angle controller started')

        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.running = False

    def stop(self):
        """Stop the controller and close connections"""
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        if self.ser and self.ser.is_open:
            self.ser.close()
            print('Arduino connection closed')

        print('Position to angle controller stopped')

    def calculate_motor_commands(self, error_x, error_y):
        """
        Convert pixel errors to motor movement commands using pinhole camera model
        and a P-controller.
        """

        # --- New Trigonometric Calculation ---
        # Calculate angular error in radians using atan(pixel_error / focal_length)
        pan_error_rad = math.atan(error_x / self.focal_length_x)
        tilt_error_rad = math.atan(error_y / self.focal_length_y)

        # Convert angular error to degrees
        pan_error_deg = math.degrees(pan_error_rad)
        tilt_error_deg = math.degrees(tilt_error_rad)

        # --- Apply P-controller ---
        # The command is proportional to the error.
        # You can tune Kp_pan and Kp_tilt for smoother/faster response.
        pan_movement = self.Kp_pan * pan_error_deg

        # Apply negative gain to tilt to match image coordinates (Y points down)
        tilt_movement = -self.Kp_tilt * tilt_error_deg

        return pan_movement, tilt_movement

    def _parse_log_entry(self, line):
        """Parse a JSON log entry and extract vector coordinates"""
        try:
            data = json.loads(line.strip())
            if (
                data.get('level') == 'INFO'
                and 'vector_x' in data
                and 'vector_y' in data
            ):
                # We assume vector_x and vector_y are the pixel errors from the center
                # (e.g., detection_x - center_x)
                return data['vector_x'], data['vector_y']
        except json.JSONDecodeError:
            pass
        return None, None

    def _send_command_to_arduino(self, pan_degrees, tilt_degrees):
        """Send motor commands to Arduino"""
        if self.ser and self.ser.is_open:
            try:
                command = f"{pan_degrees:.2f},{tilt_degrees:.2f}\n"
                self.ser.write(command.encode())
                print(
                    f"Sent command: Pan {pan_degrees:.2f}°, Tilt {tilt_degrees:.2f}°",
                )

                # Clear any pending data to prevent buffer overflow
                while self.ser.in_waiting:
                    self.ser.readline()

            except serial.SerialException as e:
                print(f"Error sending command to Arduino: {e}")

    def _run(self):
        """Main monitoring loop that reads the log file and processes new entries"""
        print(f"Monitoring log file: {self.log_file_path}")

        # Start monitoring from the end of the file
        if not self.log_file_path.exists():
            print(
                f"Log file {self.log_file_path} does not exist. Waiting for creation...",
            )

        last_position = 0
        if self.log_file_path.exists():
            last_position = self.log_file_path.stat().st_size

        while self.running:
            try:
                if self.log_file_path.exists():
                    current_size = self.log_file_path.stat().st_size

                    # Check if file has new content
                    if current_size > last_position:
                        with open(self.log_file_path) as f:
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = current_size

                        # Process new log entries
                        for line in new_lines:
                            if not self.running:
                                break

                            vector_x, vector_y = self._parse_log_entry(line)
                            if vector_x is not None and vector_y is not None:
                                # Calculate motor commands
                                pan_degrees, tilt_degrees = (
                                    self.calculate_motor_commands(
                                        vector_x, vector_y,
                                    )
                                )

                                # Send to Arduino
                                self._send_command_to_arduino(
                                    pan_degrees, tilt_degrees,
                                )

                time.sleep(0.1)  # Small delay to prevent excessive CPU usage

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)

        print('Monitoring loop stopped')
