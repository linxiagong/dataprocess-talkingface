import json
import logging
from typing import Any

import numpy as np


# Define a custom JSON encoder that can handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_dump(obj: Any, file_path):
    with open(file_path, 'w') as fp:
        json.dump(obj, fp, indent=2, separators=(',', ': '), cls=NumpyEncoder)
    logging.info(f'\t-> File {file_path} saved.')
