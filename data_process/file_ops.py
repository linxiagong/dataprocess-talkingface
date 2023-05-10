import json
from typing import Any

def json_dump(obj:Any, file_path):
    with open(file_path, 'w') as fp:
        json.dump(obj, fp, indent=2, separators=(',', ': '))