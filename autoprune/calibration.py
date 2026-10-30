"""Small input checks shared by OFT calibration readers."""
import json


def read_language(attrs, source):
    try:
        value = json.loads(attrs['problem_info'])['language_instruction']
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Calibration language is missing or malformed in {source}') from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Calibration language is empty in {source}')
    return value
