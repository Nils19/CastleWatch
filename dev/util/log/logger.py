from __future__ import annotations

import json
import logging

from dev.util.constants import DEBUG
from dev.util.constants import FILEPATH_SIMULATION_LOG


class DictJsonFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, dict):
            log_data = {'level': record.levelname}
            log_data.update(record.msg)
            return json.dumps(log_data)
        return super().format(record)


logger = logging.getLogger(__name__)

if DEBUG is True:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

handler = logging.FileHandler(FILEPATH_SIMULATION_LOG, mode='w')
formatter = DictJsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
