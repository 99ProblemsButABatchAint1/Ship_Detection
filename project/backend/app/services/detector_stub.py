from typing import List, Dict

def detect_stub() -> List[Dict]:
    # Fake result so frontend can test drawing boxes
    # Coordinates are in "image pixel" space for now (example)
    return [
        {"x": 80, "y": 60, "w": 220, "h": 140, "score": 0.9, "label": "ship"}
    ]
