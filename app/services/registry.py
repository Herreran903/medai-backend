# Este archivo define un registro de modelos disponibles para su uso en el sistema.
# Cada modelo está asociado a una clave única que permite su identificación y acceso.
# Los modelos implementan diferentes enfoques para la extracción de características.

from typing import Dict

from app.models.llm import (
    LLMExtractor,  # Importa el extractor basado en modelos de lenguaje grandes (LLM).
)
from app.models.lstm import LSTMExtractor  # Importa el extractor basado en redes LSTM.
from app.models.transformer import (
    TransformerExtractor,  # Importa el extractor basado en arquitecturas Transformer.
)

# Diccionario que actúa como un registro central de modelos.
# Las claves son cadenas que identifican el tipo de modelo, y los valores son instancias de los extractores correspondientes.
MODEL_REGISTRY: Dict[str, object] = {
    "lstm": LSTMExtractor(),  # Modelo basado en LSTM, útil para secuencias largas con dependencias temporales.
    "transformer": TransformerExtractor(),  # Modelo basado en Transformer, eficiente para tareas con atención contextual.
    "llm": LLMExtractor(),  # Modelo basado en LLM, diseñado para tareas complejas de lenguaje natural.
}
