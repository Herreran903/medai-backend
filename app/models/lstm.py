from typing import Dict, List


class LSTMExtractor:
    def __init__(self):
        self.dummy = True

    def predict(self, text: str) -> List[Dict]:
        return [
            {"type": "PEEP", "text": "PEEP 8 cmH2O", "score": 0.9},
            {"type": "RR", "text": "RR 16 rpm", "score": 0.88},
        ]
