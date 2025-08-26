from typing import Dict, List


class TransformerExtractor:
    def __init__(self):
        # No carga nada, modo demo
        self.dummy = True

    def predict(self, text: str) -> List[Dict]:
        # Devuelve entidades quemadas sin usar 'text'
        return [
            {"type": "FiO2", "text": "FiO2 60%", "score": 0.95},
            {"type": "PEEP", "text": "PEEP 8 cmH2O", "score": 0.92},
            {"type": "VT", "text": "VT 420 mL", "score": 0.90},
        ]
