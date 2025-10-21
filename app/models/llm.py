from typing import Dict, List


class LLMExtractor:
    def __init__(self):
        self.dummy = True

    def predict(self, text: str) -> List[Dict]:
        return [
            {"type": "FiO2", "text": "FiO2 60%", "score": 0.95},
            {"type": "VT", "text": "VT 420 mL", "score": 0.9},
        ]
