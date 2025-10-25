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

    def __init__(self, camera_id, width, height):
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
        self._target_interval = 0.2
        self._width = width
        self._height = height

        # Color bounds (default to orange detection)
        self.lower_color = np.array([5, 120, 150])
        self.upper_color = np.array([35, 255, 255])

        # Threading and control
        self.detection_thread = None
        self.logging_thread = None
        self.running = False
        self.thread_lock = threading.Lock()

        # Detection results
        self.current_vector = None
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

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

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

            vector = self._process_frame(frame)

            # Thread-safe update of detection results
            with self.thread_lock:
                self.current_vector = vector
                if vector is not None:
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
                vector_to_log = self.current_vector

            # Log only when object is detected
            if vector_to_log is not None:
                log_data = {
                    'vector_x': vector_to_log[0],
                    'vector_y': vector_to_log[1],
                }
                logger.info(log_data)
                
                # Force flush to ensure logs appear immediately in tail -f
                for handler in logger.handlers:
                    if hasattr(handler, 'flush'):
                        handler.flush()

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
            tuple: (vector_x, vector_y) from camera center to object center if detected, None otherwise
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

        vector = None
        object_center = None
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

                # Calculate vector from camera center to object center
                camera_center_x = self._width // 2
                camera_center_y = self._height // 2
                vector_x = center_x - camera_center_x
                vector_y = center_y - camera_center_y

                vector = (vector_x, vector_y)
                object_center = (center_x, center_y)

        # Add visual overlays
        self._draw_overlays(frame, object_center, vector)

        # Debug windows
        if DEBUG:
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            cv2.waitKey(1)

        return vector

    def _draw_overlays(self, frame, object_center, vector):
        """
        Draw visual overlays on the frame including camera center, object center, and vector.

        Args:
            frame: BGR frame to draw on
            object_center: (x, y) coordinates of detected object center, or None
            vector: (vector_x, vector_y) from camera center to object, or None
        """
        camera_center_x = self._width // 2
        camera_center_y = self._height // 2
        camera_center = (camera_center_x, camera_center_y)

        # Draw camera center as a blue cross
        cv2.drawMarker(frame, camera_center, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, 'Camera Center', (camera_center_x + 15, camera_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # If object is detected, draw object center and vector
        if object_center is not None and vector is not None:
            # Draw object center as a green circle
            cv2.circle(frame, object_center, 8, (0, 255, 0), -1)
            cv2.putText(frame, 'Object Center', (object_center[0] + 15, object_center[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw vector line from camera center to object center
            cv2.arrowedLine(frame, camera_center, object_center, (0, 255, 255), 2, tipLength=0.1)

            # Display vector values
            vector_text = f'Vector: ({vector[0]}, {vector[1]})'
            cv2.putText(frame, vector_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
