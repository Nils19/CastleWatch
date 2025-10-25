"""
ObjectDetector Class for CastleWatch
Modular color-based object detection with 60Hz logging
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from dev.util.constants import DEBUG
from dev.util.log.logger import logger


class ColorObjectDetector:
    """
    A modular ObjectDetector class that detects colored objects in video feed,
    calculates their center coordinates, and logs positions at 60Hz frequency.
    """

    def __init__(self, camera_id):
        """
        Initialize the ObjectDetector with camera configuration.

        Args:
            camera_id (int): Camera index for cv2.VideoCapture
        """
        # Clear log file
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                open(handler.baseFilename, 'w').close()
                break

        self.camera_id = camera_id
        self.cap = None
        self.kernel = np.ones((5, 5), np.uint8)
        self._target_interval = 1 / 30

        # Color bounds (default to orange detection)
        self.lower_color = np.array([5, 120, 150])
        self.upper_color = np.array([35, 255, 255])

        # Threading and control
        self.detection_thread = None
        self.logging_thread = None
        self.running = False
        self.thread_lock = threading.Lock()

        # Detection results
        self.current_center = None
        self.last_detection_time = 0

        # Initialize camera
        self._setup_camera()

    def _setup_camera(self):
        """Initialize camera capture with 640x480 resolution."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera with ID {self.camera_id}",
            )

        # Set resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def set_color_bounds(self, lower_color, upper_color):
        """
        Configure HSV color range for object detection.

        Args:
            lower_color (numpy.array): HSV lower bound (e.g., [5, 120, 150])
            upper_color (numpy.array): HSV upper bound (e.g., [35, 255, 255])
        """
        self.lower_color = np.array(lower_color)
        self.upper_color = np.array(upper_color)

    def start(self):
        """
        Start object detection on separate threads (non-blocking).
        Starts both frame processing and 60Hz logging threads.
        """
        if self.running:
            return

        self.running = True

        # Start detection thread (processes frames as fast as possible)
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
        )
        self.detection_thread.start()

        # Start logging thread (exactly 60Hz)
        self.logging_thread = threading.Thread(
            target=self._logging_loop,
            daemon=True,
        )
        self.logging_thread.start()

    def stop(self):
        """Stop the detection thread and cleanup resources."""
        self.running = False

        # Wait for threads to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)

        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=1.0)

        # Release camera resources
        if self.cap:
            self.cap.release()

        # Close any OpenCV windows
        cv2.destroyAllWindows()

    def _detection_loop(self):
        """
        Main detection loop that processes camera frames as fast as possible.
        Runs in a separate thread.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            center = self._process_frame(frame)

            # Thread-safe update of detection results
            with self.thread_lock:
                self.current_center = center
                if center is not None:
                    self.last_detection_time = time.time()

    def _logging_loop(self):
        """
        Logging loop that runs at exactly 60Hz (every 16.67ms).
        Logs detection results independently of frame processing speed.
        """
        while self.running:
            start_time = time.time()

            # Get current detection result (thread-safe)
            with self.thread_lock:
                center_to_log = self.current_center

            # Log only when object is detected
            if center_to_log is not None:
                log_data = {
                    'center_x': center_to_log[0],
                    'center_y': center_to_log[1],
                }
                logger.info(log_data)

            # Calculate sleep time to maintain 60Hz
            elapsed = time.time() - start_time
            sleep_time = self._target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self, frame):
        """
        Process a single frame for object detection.

        Args:
            frame: Input BGR frame from camera

        Returns:
            tuple: (center_x, center_y) if object detected, None otherwise
        """
        # Apply Gaussian blur (3x3 kernel)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create color mask using cv2.inRange
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # Morphological operations
        # Opening: erode → dilate (2 iterations each)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        # Closing: dilate → erode (2 iterations each)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.erode(mask, self.kernel, iterations=2)

        # Find contours using cv2.findContours with RETR_EXTERNAL
        contours = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = contours[0] if len(contours) == 2 else contours[1]

        center = None
        if len(contours) > 0:
            # Select largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate center coordinates using moments
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])

                # Apply minimum radius threshold (radius > 5)
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

                if radius > 5:
                    center = (center_x, center_y)

        # Debug windows
        if DEBUG:
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            cv2.waitKey(1)

        return center
