from typing import Dict

from app.models.llm import LLMExtractor
from app.models.lstm import LSTMExtractor
from app.models.transformer import TransformerExtractor

MODEL_REGISTRY: Dict[str, object] = {
    "lstm": LSTMExtractor(),
    "transformer": TransformerExtractor(),
    "llm": LLMExtractor(),
}
