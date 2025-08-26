# app/models/lstm_extractor.py
from typing import Dict, List


class LSTMExtractor:
    def __init__(self):
        # por ahora no cargamos nada (dummy)
        self.dummy = True

    def predict(self, text: str) -> List[Dict]:
        # respuesta fija de ejemplo
        return [
            {"type": "PEEP", "text": "PEEP 8 cmH2O", "score": 0.9},
            {"type": "RR", "text": "RR 16 rpm", "score": 0.88},
        ]
