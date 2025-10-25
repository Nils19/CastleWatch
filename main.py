from __future__ import annotations

import time

from dev.control.position_to_angle_controller import PositionToAngleController
from dev.cv.color_object_detector import ColorObjectDetector

detector = ColorObjectDetector(camera_id=2, width=640, height=480)
controller = PositionToAngleController()
detector.start()
controller.start()

while True:
    try:
        # keep main thread alive
        time.sleep(1)
    except KeyboardInterrupt:
        detector.stop()
        controller.stop()
        exit()
