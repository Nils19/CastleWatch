from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

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
        # --- PID Gains (These require tuning!) ---
        # kp_pan=0.2,   # Proportional gain
        # ki_pan=0.05,  # Integral gain
        # kd_pan=0.1,   # Derivative gain
        # kp_tilt=0.2,  # Proportional gain
        # ki_tilt=0.05, # Integral gain
        # kd_tilt=0.3,  # Derivative gain
        kp_pan=0.35,  # Drastically reduced Kp to stop over-reaction
        ki_pan=0.01,  # Very small Ki to prevent integral windup
        kd_pan=0.01,  # Reduced Kd to prevent "derivative kick"
        kp_tilt=0.35,  # Match pan values for tilt
        ki_tilt=0.01,
        kd_tilt=0.01,
    ):
        """Initialize the controller with Arduino and log file settings"""
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.log_file_path = Path(log_file_path)
        self.ser = None
        self.running = False
        self.thread = None

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.focal_length_x = (self.frame_width / 2) / math.tan(
            math.radians(camera_fov_horizontal / 2),
        )
        self.focal_length_y = (self.frame_height / 2) / math.tan(
            math.radians(camera_fov_vertical / 2),
        )
        print(
            f"Calculated focal lengths: fx={self.focal_length_x:.2f}px, fy={self.focal_length_y:.2f}px",
        )

        # --- Store PID gains ---
        self.Kp_pan, self.Ki_pan, self.Kd_pan = kp_pan, ki_pan, kd_pan
        self.Kp_tilt, self.Ki_tilt, self.Kd_tilt = kp_tilt, ki_tilt, kd_tilt

        # --- PID state variables ---
        self._previous_time = None
        self._previous_error_pan, self._previous_error_tilt = 0.0, 0.0
        self._integral_pan, self._integral_tilt = 0.0, 0.0

        # --- Anti-windup clamp for the integral term ---
        self.integral_clamp_pan = 10.0  # Max integral value in degrees
        self.integral_clamp_tilt = 10.0  # Max integral value in degrees

    def start(self):
        """Start monitoring the log file and sending commands to Arduino"""
        if self.running:
            print('Controller is already running')
            return

        try:
            self.ser = serial.Serial(
                self.arduino_port,
                self.baud_rate,
                timeout=1,
            )
            time.sleep(2)
            print(f"Connected to Arduino on {self.arduino_port}")

            # --- Reset PID state on start ---
            self.reset_pid()

            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print('Position to angle PID controller started')

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

    def reset_pid(self):
        """Resets the PID controller's state variables."""
        print('Resetting PID state.')
        self._previous_time = time.time()
        self._previous_error_pan, self._previous_error_tilt = 0.0, 0.0
        self._integral_pan, self._integral_tilt = 0.0, 0.0

    def calculate_pid_commands(self, error_x, error_y):
        """
        Convert pixel errors to motor movement commands using a PID controller.
        """
        current_time = time.time()
        dt = current_time - self._previous_time
        # Avoid division by zero on the first loop
        if dt <= 0.001:
            dt = 0.001

        # --- 1. Calculate Angular Error (Proportional Term) ---
        pan_error_rad = math.atan(error_x / self.focal_length_x)
        tilt_error_rad = math.atan(error_y / self.focal_length_y)
        pan_error_deg = math.degrees(pan_error_rad)
        tilt_error_deg = math.degrees(tilt_error_rad)

        # --- 2. Calculate Integral Term (with Anti-Windup) ---
        self._integral_pan += pan_error_deg * dt
        self._integral_tilt += tilt_error_deg * dt

        # Clamp the integral to prevent windup
        self._integral_pan = max(
            min(self._integral_pan, self.integral_clamp_pan),
            -self.integral_clamp_pan,
        )
        self._integral_tilt = max(
            min(self._integral_tilt, self.integral_clamp_tilt),
            -self.integral_clamp_tilt,
        )

        # --- 3. Calculate Derivative Term ---
        derivative_pan = (pan_error_deg - self._previous_error_pan) / dt
        derivative_tilt = (pan_error_deg - self._previous_error_tilt) / dt

        # --- 4. Combine PID terms to get the output ---
        pan_movement = (
            (self.Kp_pan * pan_error_deg)
            + (self.Ki_pan * self._integral_pan)
            + (self.Kd_pan * derivative_pan)
        )

        # Apply negative gain to tilt to match image coordinates (Y points down)
        tilt_movement = -(
            (self.Kp_tilt * tilt_error_deg)
            + (self.Ki_tilt * self._integral_tilt)
            + (self.Kd_tilt * derivative_tilt)
        )

        # --- 5. Update state for the next iteration ---
        self._previous_error_pan = pan_error_deg
        self._previous_error_tilt = tilt_error_deg
        self._previous_time = current_time

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
                while self.ser.in_waiting:
                    self.ser.readline()
            except serial.SerialException as e:
                print(f"Error sending command to Arduino: {e}")

    def _run(self):
        """Main monitoring loop"""
        print(f"Monitoring log file: {self.log_file_path}")
        if not self.log_file_path.exists():
            print(f"Log file {self.log_file_path} does not exist. Waiting...")

        last_position = (
            self.log_file_path.stat().st_size
            if self.log_file_path.exists()
            else 0
        )

        # --- ADDED: Timeout for resetting PID if no new data arrives ---
        last_detection_time = time.time()
        no_detection_timeout = 1.0  # seconds

        while self.running:
            try:
                if self.log_file_path.exists():
                    current_size = self.log_file_path.stat().st_size
                    if current_size > last_position:
                        with open(self.log_file_path) as f:
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = current_size

                        for line in new_lines:
                            if not self.running:
                                break
                            vector_x, vector_y = self._parse_log_entry(line)
                            if vector_x is not None and vector_y is not None:
                                # --- Use the new PID calculation method ---
                                pan_degrees, tilt_degrees = (
                                    self.calculate_pid_commands(
                                        vector_x, vector_y,
                                    )
                                )
                                self._send_command_to_arduino(
                                    pan_degrees, tilt_degrees,
                                )
                                last_detection_time = time.time()  # Reset timer

                    # --- If no detection for a while, reset PID to prevent stale values ---
                    if time.time() - last_detection_time > no_detection_timeout:
                        self.reset_pid()
                        # Keep the timer updated to avoid repeated resets
                        last_detection_time = time.time()

                time.sleep(0.02)  # Reduced delay for more responsive control

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)
        print('Monitoring loop stopped')
