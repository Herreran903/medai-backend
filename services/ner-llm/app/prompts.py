# =============================================================================
# PROMPTS PARA EVALUACION DE LLMs EN NER
# =============================================================================
# Archivo centralizado con los prompts de sistema y extraccion.
# Modificar aqui para cambiar el comportamiento del LLM sin tocar el notebook.

import json
from pathlib import Path
from typing import List, Optional

# Ruta por defecto del archivo de ejemplos few-shot
_DEFAULT_FEW_SHOT_PATH = (
    Path(__file__).resolve().parent / "examples" / "few_shot_examples.json"
)


def load_few_shot_examples(path: Optional[Path] = None) -> List[dict]:
    """Carga ejemplos few-shot desde JSON generado por data/scripts/generate_few_shot_examples.py."""
    path = Path(path) if path else _DEFAULT_FEW_SHOT_PATH
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("examples", [])


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = (
    "Actuas exclusivamente como un extractor de entidades para evaluacion experimental de NER. "
    "Recibiras textos ya preprocesados dentro de un pipeline comparativo que incluye BiLSTM y modelos fine-tuned; "
    "por tanto, debes producir unicamente una salida estructurada estrictamente valida conforme al esquema Pydantic/JSON Schema proporcionado, "
    "sin texto libre, sin explicaciones y sin encabezados. "
    "La inferencia asume decodificacion guiada estricta (JSON Schema / FSM / XGrammar), por lo que solo estan permitidos tokens compatibles con el esquema. "
    "Extrae todas las entidades presentes en el texto, indicando texto exacto, etiqueta y offsets de caracteres (start, end) alineados con el texto de entrada. "
    "No inventes entidades, no cambies etiquetas, no omitas campos y no agregues claves adicionales. "
    "Si existe ambiguedad, prioriza recall sin violar el esquema."
)


# =============================================================================
# EXTRACTION PROMPT (prompt de usuario / instrucciones de extraccion)
# =============================================================================

EXTRACTION_PROMPT = """Eres un experto en extraccion de entidades clinicas de notas medicas de pacientes en ventilacion mecanica.

Tu tarea es extraer UNICAMENTE las entidades clinicas mencionadas en el texto.

IMPORTANTE - REGLAS DE EXTRACCION:
1. SIEMPRE incluir la etiqueta/sigla cuando este presente en el texto junto al valor
2. Ejemplo: "FiO2 60%" -> extraer "FiO2 60%" (incluir la etiqueta "FiO2")
3. Ejemplo: "PEEP 8 cmH2O" -> extraer "PEEP 8 cmH2O" (incluir la etiqueta "PEEP")
4. Ejemplo: "VT 420 ml" -> extraer "VT 420 ml" (incluir la etiqueta "VT")
5. Ejemplo: "modo AC VC" -> extraer "AC VC" (NO incluir la palabra descriptiva "modo")
6. Si SOLO aparece el valor sin etiqueta en el texto, extraer solo el valor

**CONFIGURACION DE VENTILACION:**
- MODO: Modo del ventilador (sin palabras descriptivas como "modo", "en modo", "VMI")
  Variantes: AC, VC, PC, PC+, VC+, SIMV, PSV, CPAP, PRVC, ACV, VCV, PCV, BiPAP
  Ejemplos: "modo AC VC" -> extraer "AC VC", "en modo PC+" -> extraer "PC+", "VMI modo SIMV" -> extraer "SIMV"

- FIO2: Fraccion inspirada de oxigeno (incluir etiqueta + valor)
  Variantes: FiO2, FIO2, fio2, Fi02, fraccion inspirada de oxigeno
  Ejemplos: "FiO2 40%", "fio2: 0.5", "FIO2 80%", "Fi02 100"

- PEEP: Presion positiva al final de la espiracion
  Variantes: PEEP, peep, Peep
  Ejemplos: "PEEP 8", "peep: 10", "PEEP 12 cmH2O"

- FR: Frecuencia respiratoria
  Variantes: FR, fr, Freq resp, frecuencia respiratoria, rpm
  Ejemplos: "FR 14/20", "fr: 18 rpm", "FR 20", "frecuencia respiratoria 22"

- VT: Volumen tidal/corriente
  Variantes: VT, Vt, vt, Vol, vol, VC (volumen corriente), volumen tidal
  Ejemplos: "VT 380", "Vt: 450 mL", "Vol 480/490", "vol corriente 420"

- FLUJO: Flujo inspiratorio
  Variantes: Flujo, flujo, flow
  Ejemplos: "Flujo 45", "flujo 50 L/min"

- I_E: Relacion inspiracion:espiracion
  Variantes: I:E, I/E, relacion I:E, Rel I:E, RIE
  Ejemplos: "I:E 1:2", "1:2", "relacion 1:3", "Rel. 1:1.5"

- SENS: Sensibilidad del trigger
  Variantes: Sens, sens, sensibilidad, trigger
  Ejemplos: "Sens 2", "sensibilidad: 3", "sens: 1.5"

**RESPUESTA A LA VENTILACION:**
- SAO2: Saturacion de oxigeno
  Variantes: SaO2, SAO2, Sat, sat, saturacion, SO2, SpO2, SatO2
  Ejemplos: "SaO2 95%", "Sat 92%", "saturacion 97%", "SO2 94%", "SpO2 98"

- PP: Presion pico de via aerea
  Variantes: PP, Ppico, P pico, presion pico, peak pressure
  Ejemplos: "PP 25", "Ppico: 22", "presion pico 28", "P pico 30"

- PMES: Presion meseta/plateau
  Variantes: Pmes, Pmeseta, P meseta, plateau, presion meseta, P plateau
  Ejemplos: "Pmeseta 18", "plateau 20", "P meseta: 22", "presion plateau 19"

- PM: Poder mecanico
  Variantes: PM, poder mecanico, mechanical power

**ANTROPOMETRICOS:**
- EDAD: Edad del paciente - SOLO valor y unidad
  Variantes: anos, anos de edad, a, years
  Correcto: "78 anos", "65 anos", "42 a"
  Incorrecto: "edad 78 anos", "paciente de 65 anos de edad", "Edad: 70"

- PESO: Peso corporal - SOLO valor y unidad, NO la palabra "peso"
  Variantes: kg, kilos, kilogramos, Kg, KG
  Correcto: "65 kg", "72 kg", "95", "80 kilos"
  Incorrecto: "Peso 65 kg", "peso: 72 kg", "Peso(Kg): 70"

**SIGNOS VITALES:**
- TEMP: Temperatura corporal
  Variantes: T, Temp, temperatura, temp max, temp min, celsius
  Ejemplos: "T 36.5", "Temp: 37.2", "36.8 C", "T 37C", "temperatura 38.5"

- PA: Presion arterial completa (sistolica/diastolica)
  Variantes: PA, TA, presion arterial, tension arterial, PA(mmHg)
  Ejemplos: "PA 120/80", "TA: 130/85", "TA 110/70 mmHg", "presion arterial 125/80"

- PAS: Presion arterial sistolica SOLAMENTE
  Variantes: PAS, TAS, presion sistolica, tension sistolica, sistolica
  Ejemplos: "TAS 125", "PAS 140", "TAS menor 150", "sistolica 130"

- PAM: Presion arterial media
  Variantes: PAM, TAM, presion arterial media, tension arterial media, MAP
  Ejemplos: "PAM 85", "TAM: 78", "PAM 65 mmHg", "TAM 70"

- FC: Frecuencia cardiaca
  Variantes: FC, fc, frecuencia cardiaca, pulso, lpm, lat/min, latidos
  Ejemplos: "FC 82", "FC: 75 lpm", "fc 90", "frecuencia cardiaca 88"

- GLICEMIA: Glucosa/glucometria en sangre
  Variantes: Glicemia, glicemia, glucometria, glucosa, HGT, dextrostix, mg/dL
  Ejemplos: "Glicemia 120", "glucometria: 145", "125-130-140 mg/dL", "HGT 110"

- POSTURA: Posicion/posicionamiento del paciente
  Variantes: posicion, postura, decubito, cabecera, fowler, semifowler, prono, supino
  Ejemplos: "Cabecera 30", "supino", "prono", "decubito lateral", "semifowler", "Cabecera 45 grados", "posicion fowler"

**DIAGNOSTICOS (DX):**
- DX: Diagnosticos clinicos - extraer SOLO el nucleo diagnostico (2-5 palabras max)
  Variantes: diagnostico, Dx, dx, impresion diagnostica
  Correcto: "SDRA severo", "Falla ventilatoria", "Shock septico", "NAV", "Neumonia asociada a ventilador"
  Correcto: "HTA", "DM2", "EPOC", "IRC", "ERC", "IAM" (siglas de diagnosticos)
  Incorrecto: "paciente con antecedente de falla ventilatoria tipo 2" (muy largo)
  Incorrecto: "se considera SDRA severo por criterios de Berlin" (incluye explicacion)

**GASES ARTERIALES:**
- PH: pH sanguineo
  Variantes: pH, ph, PH
  Ejemplos: "pH 7.35", "ph: 7.42", "PH 7.28"

- PACO2: Presion parcial de CO2
  Variantes: PaCO2, PCO2, pCO2, CO2, paco2
  Ejemplos: "PaCO2 42", "pCO2: 38", "PCO2 45", "CO2 40"

- HCO3: Bicarbonato
  Variantes: HCO3, HCO3-, bicarbonato, Bic
  Ejemplos: "HCO3 24", "bicarbonato: 22", "HCO3- 26", "Bic 23"

- BE: Exceso de base
  Variantes: BE, EB, exceso de base, base excess
  Ejemplos: "BE -2", "EB: +1", "BE 3.5", "exceso de base -4"

- PAO2: Presion parcial de O2
  Variantes: PaO2, PO2, pO2, pao2
  Ejemplos: "PaO2 85", "pO2: 92", "PO2 78"

- PAFI: Relacion PaO2/FiO2
  Variantes: PAFI, PaFi, PaFiO2, pafi, Pa/Fi, relacion PaO2/FiO2, indice de Kirby
  Ejemplos: "PaFi 180", "PAFI: 250", "pafi 120", "Pa/Fi 200"

**REGLAS CRITICAS:**
1. NO incluir palabras descriptivas como "modo", "peso", "edad", "temperatura" antes del valor
2. SI incluir siglas medicas (FiO2, PEEP, FC, TAM, etc.) cuando estan junto al valor numerico
3. Para DX: extraer solo el diagnostico, no la oracion completa
4. Buscar TODAS las variantes de cada entidad listadas arriba
5. Extraer TODAS las entidades que encuentres, priorizar recall sobre precision
6. OBLIGATORIO: Calcular start y end (offsets de caracteres) para cada entidad
   - start: posicion del primer caracter de la entidad en el texto (0-indexed)
   - end: posicion del caracter DESPUES del ultimo caracter de la entidad
   - Ejemplo: en "Peso 65 kg", si extraes "65 kg", start=5, end=10

Texto a analizar:
{text}

Extrae las entidades en formato JSON estructurado."""


# =============================================================================
# FUNCIONES DE FORMATEO DE PROMPTS
# =============================================================================


def _format_few_shot_examples(few_shot_examples: List[dict]) -> str:
    """Convierte ejemplos few-shot (texto + entidades) en bloque de prompt."""
    blocks = []
    for idx, ex in enumerate(few_shot_examples or [], start=1):
        if not isinstance(ex, dict):
            continue
        text = str(ex.get("text") or "").strip()
        entities = ex.get("entities") or []
        if not text:
            continue

        normalized_entities = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            etype = str(ent.get("type") or "").strip().upper()
            etext = str(ent.get("text") or "").strip()
            if not etype or not etext:
                continue

            # Si el ejemplo no trae offsets, se estiman para mantener formato consistente.
            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None:
                hit = text.find(etext)
                if hit >= 0:
                    start = hit
                    end = hit + len(etext)

            if start is None or end is None:
                continue

            normalized_entities.append(
                {"type": etype, "text": etext, "start": int(start), "end": int(end)}
            )

        blocks.append(
            "Ejemplo few-shot "
            + str(idx)
            + ":\nTexto:\n"
            + text
            + "\nSalida esperada JSON:\n"
            + json.dumps({"entities": normalized_entities}, ensure_ascii=False)
        )

    return "\n\n".join(blocks)


def build_user_prompt(
    text: str,
    entity_types: List[str],
    schema: dict,
    *,
    include_schema: bool = True,
    prompt_mode: str = "zero_shot",
    few_shot_examples: Optional[List[dict]] = None,
) -> str:
    """Construye el prompt de usuario para extraccion de entidades."""
    labels = ", ".join(entity_types) if entity_types else "(sin etiquetas)"
    schema_block = ""
    if include_schema:
        schema_json = json.dumps(schema, ensure_ascii=False)
        schema_block = "Esquema JSON estricto:\n" + schema_json + "\n\n"

    prompt_mode = str(prompt_mode or "zero_shot").strip().lower()
    few_shot_examples = few_shot_examples or []

    few_shot_block = ""
    if prompt_mode == "few_shot":
        few_shot_rendered = _format_few_shot_examples(few_shot_examples)
        if few_shot_rendered:
            few_shot_block = (
                "\n\nEjemplos de referencia (few-shot):\n"
                + few_shot_rendered
                + "\n\nAhora extrae entidades del siguiente texto."
            )

    base_prompt = EXTRACTION_PROMPT.format(text=text) + few_shot_block
    return (
        base_prompt
        + "\n\n"
        + "Modo de prompting: "
        + prompt_mode
        + "\n"
        + "Etiquetas permitidas: "
        + labels
        + "\n"
        + "Incluye SIEMPRE la clave 'entities'. Si no hay entidades, devuelve []\n"
        + "No incluyas texto explicativo fuera del JSON.\n"
        + schema_block
    )
