from typing import Dict, Any
import json


def read_json(filename: str) -> Dict[Any, Any]:
    with open(filename) as f:
        return json.load(f)


def write_json(json_object: Dict[Any, Any], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(json_object, f)
