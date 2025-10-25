from __future__ import annotations

import time

from dev.cv.color_object_detector import ColorObjectDetector

detector = ColorObjectDetector(camera_id=1)
detector.start()

while True:
    try:
        # keep main thread alive
        time.sleep(1)
    except KeyboardInterrupt:
        detector.stop()
        exit()
