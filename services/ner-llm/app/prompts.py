# =============================================================================
# PROMPTS PARA EVALUACION DE LLMs EN NER
# =============================================================================
# Archivo centralizado con los prompts de sistema y extraccion.
# Modificar aqui para cambiar el comportamiento del LLM sin tocar el notebook.

import json
from typing import List

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

REGLAS GENERALES:
1. Anota solo menciones textuales explicitas.
2. Si una entidad aparece varias veces y cada aparicion es valida, anota todas sus menciones.
3. Cuando ayude a identificar mejor la entidad, conserva la mencion mas completa.
4. En entidades numericas, no anotes solo el numero si el texto incluye tambien el nombre del parametro, su sigla o la unidad. Prefiere spans completos como "FC 96 lpm", "FiO2 40 %", "TA 128/63 mmHg", "pH 7.32".
5. No extiendas el span a texto adicional que no haga parte de la entidad.
6. No anotes metas terapeuticas, rangos objetivo, intervalos de referencia ni umbrales deseados. Solo mediciones reales del paciente.
7. No anotes duplicados artificiales ni fragmentos incompletos si existe una mencion completa claramente preferible.

ENTIDADES QUE NO SE ANOTAN: PAS, PAD. Nunca uses estas etiquetas.

**CONFIGURACION DE VENTILACION:**
- MODO: Modo del ventilador.
  Si aparece la palabra "modo", intentar conservarla dentro del span.
  Si el texto solo trae la sigla del modo sin la palabra "modo", anotar la sigla sola.
  Si el modo aparece en forma descriptiva, anotar la formulacion completa relevante.
  No extender el span a parametros adicionales pegados que no hagan parte del modo.
  Variantes: AC, VC, PC, PC+, VC+, SIMV, PSV, CPAP, PRVC, ACV, VCV, PCV, BiPAP
  Correcto: "modo AC VC", "PC+", "VMI modo SIMV", "controlado por volumen", "modo VC", "PRVC", "ACV"
  Incorrecto: "modo VC:380" completo como un solo span si ":380" corresponde a otro parametro

- FIO2: Fraccion inspirada de oxigeno
  Variantes: FiO2, FIO2, fio2, Fi02, fraccion inspirada de oxigeno
  Correcto: "FiO2 40%", "fio2: 0.5", "FIO2 80%", "Fi02 100", "FiO2 40 %"

- PEEP: Presion positiva al final de la espiracion
  Variantes: PEEP, peep, Peep
  Correcto: "PEEP 8", "peep: 10", "PEEP 12 cmH2O"

- FR: Frecuencia respiratoria
  Variantes: FR, fr, freq resp, frecuencia respiratoria, rpm
  Correcto: "FR 14/20", "fr: 18 rpm", "FR 20", "frecuencia respiratoria 22", "FR 18"

- VT: Volumen tidal o volumen corriente
  Variantes: VT, Vt, vt, Vol, vol, volumen tidal, volumen corriente
  Correcto: "VT 380", "Vt: 450 mL", "Vol 480/490", "vol corriente 420"

- I_E: Relacion inspiracion:espiracion
  Anotar cuando el texto identifique claramente la relacion. Conservar la forma completa.
  No anotar fragmentos ambiguos o numeros sueltos que no expresen claramente la relacion.
  Variantes: I:E, I/E, relacion I:E, Rel I:E, RIE, IE
  Correcto: "I:E 1:2", "IE:1:2.7", "Relacion I:E 1:3", "Rel. 1:1.5"
  Incorrecto: "2.0" aislado, "IE 1.20" si no queda clara la relacion, fragmentos truncados como "I \"

**RESPUESTA A LA VENTILACION:**
- SAO2: Saturacion de oxigeno del paciente. No anotar saturaciones venosas (SvO2).
  Variantes: SaO2, SAO2, Sat, sat, saturacion, SO2, SpO2, SatO2
  Correcto: "SaO2 95%", "Sat 92%", "saturacion 97%", "SO2 94%", "SpO2 98", "SatO2 97 %"
  Incorrecto: "SvO2 73%", "saturaciones por encima de 90%" (meta)

**ANTROPOMETRICOS:**
- EDAD: Edad del paciente tal como aparece escrita en la nota.
  Normalmente numero + "anos". No exigir palabra introductoria fija.
  No mezclar con edad al diagnostico o codificacion diagnostica.
  Variantes: anos, anos de edad, a, years
  Correcto: "78 anos", "65 anos", "42 a", "paciente de 64 anos", "femenina de 35 anos"
  Incorrecto: "edad avanzada", "adulto mayor", "Edad al diagnostico: 61 anos"

- PESO: Peso corporal
  Variantes: peso, kg, kilos, kilogramos, Kg, KG
  Correcto: "Peso : 60 Kg", "60 kg", "Peso aprox : 100 kg", "Peso: 90 kg"

- TALLA: Estatura del paciente
  Variantes: talla, cm, m, metros, centimetros, estatura
  Correcto: "Talla 170 cm", "1.72 m", "estatura: 165 cm"

**SIGNOS VITALES:**
- TEMP: Temperatura corporal
  Variantes: T, Temp, temperatura, temp max, temp min, celsius
  Correcto: "T 36.5", "Temp: 37.2", "36.8 C", "T 37C", "temperatura 38.5"

- PA: Presion arterial completa. Anotar la mencion completa incluyendo sistolica/diastolica y unidad.
  No separar sistolica y diastolica en etiquetas distintas. No anotar metas o rangos objetivo.
  Variantes: TA, PA, presion arterial, tension arterial
  Correcto: "TA 128/63 mmHg", "PA 110/70", "TA: 130/80", "TA 120/63 mmHg"
  Incorrecto: "TAS < 140 mmHg" (meta), separar en PAS y PAD

- PAM: Presion arterial media
  Variantes: PAM, TAM, presion arterial media, tension arterial media, MAP
  Correcto: "PAM 85", "TAM: 78", "PAM 65 mmHg", "TAM 84 mmHg"

- FC: Frecuencia cardiaca
  Variantes: FC, fc, frecuencia cardiaca, pulso, lpm, lat/min, latidos
  Correcto: "FC 82", "FC: 75 lpm", "fc 90", "frecuencia cardiaca 88", "FC 96 lpm"

- GLICEMIA: Glucosa o glucometria en sangre.
  Anotar valores realmente registrados del paciente, incluidos valores puntuales o series reales.
  No anotar metas como "entre 140 y 180 mg/dl" o "por debajo de 200 mg/dl".
  Variantes: glicemia, glucometria, glucosa, HGT, dextrostix, mg/dL
  Correcto: "Glicemia 120", "glucometria: 145", "125-130-140 mg/dL", "HGT 110", "Glucometrias: 125 - 110 mg/dl"
  Incorrecto: "entre 140 y 180 mg/dl" (meta), "por debajo de 200 mg/dl" (meta)

- POSTURA: Posicion o posicionamiento del paciente.
  Conservar la formulacion mas completa disponible.
  Incluir tanto referencias a posicion como a cabecera cuando hagan parte clara de la postura.
  Variantes: posicion, postura, decubito, cabecera, fowler, semifowler, prono, supino, semisentado
  Correcto: "Cabecera 30", "supino", "prono", "decubito lateral", "semifowler", "Cabecera 45 grados", "cabecera a 30 grados", "semisentado"

**DIAGNOSTICOS:**
- DX: Diagnostico, problema clinico, comorbilidad o antecedente relevante mencionado en la nota.
  En listados estructurados, conservar la formulacion diagnostica completa cuando sea clara.
  En prosa narrativa, anotar la mencion diagnostica principal sin arrastrar informacion extra.
  Variantes: diagnostico, Dx, dx, impresion diagnostica, antecedentes, comorbilidades
  Correcto: "INSUFICIENCIA RESPIRATORIA AGUDA", "Shock septico", "Pancreatitis aguda", "SDRA severo", "neumonia asociada a ventilacion mecanica", "HTA", "DIABETES MELLITUS", "Enfermedad renal cronica agudizada"
  Incorrecto: "TEP descartado" (descartado), "posible sepsis" (hipotesis no confirmada), "SARS-CoV-2" aislado sin formulacion diagnostica, "laparotomia exploratoria" (procedimiento)

  Reglas para DX:
  1. No anotar diagnosticos descartados.
  2. No anotar hipotesis diagnosticas no confirmadas cuando el texto las presenta explicitamente como posibilidad.
  3. No anotar procedimientos quirurgicos si no constituyen un diagnostico afirmado.
  4. No tratar microorganismos aislados como DX si no aparecen dentro de una formulacion diagnostica explicita.
  5. No agregar explicaciones narrativas ni texto accesorio fuera de la mencion diagnostica.

**GASES ARTERIALES:**
  Reglas comunes: anotar la mencion completa cuando parametro y valor aparezcan juntos.
  No anotar gases venosos ni contextos claramente venosos (encabezados como "gases venosos", "GV", "venosos").
  No anotar metas, rangos objetivo ni intervalos de referencia.
  No anotar numeros aislados si el parametro no esta identificado claramente.

- PH: pH sanguineo arterial
  Variantes: pH, ph, PH
  Correcto: "pH 7.35", "ph: 7.42", "PH 7.28", "pH 7.31"

- PACO2: Presion parcial de CO2 arterial
  Variantes: PaCO2, PCO2, pCO2, CO2, paco2
  Correcto: "PaCO2 42", "pCO2: 38", "PCO2 45", "PaCO2 41 mmHg"
  Incorrecto: "PvCO2 45" (venoso), "CO2 35-40" si es meta o rango deseado

- HCO3: Bicarbonato
  Variantes: HCO3, HCO3-, bicarbonato, Bic
  Correcto: "HCO3 24", "bicarbonato: 22", "HCO3- 26", "HCO3 21"

- BE: Exceso de base
  Variantes: BE, EB, exceso de base, base excess
  Correcto: "BE -2", "EB: +1", "BE 3.5", "exceso de base -4", "BE -4"

- PAO2: Presion parcial de O2 arterial
  Variantes: PaO2, PO2, pO2, pao2
  Correcto: "PaO2 85", "pO2: 92", "PO2 78", "PaO2 62 mmHg"

- PAFI: Relacion PaO2/FiO2
  Variantes: PAFI, PaFi, PaFiO2, pafi, Pa/Fi, relacion PaO2/FiO2, indice de Kirby
  Correcto: "PaFi 180", "PAFI: 250", "pafi 120", "Pa/Fi 200", "PAFI 152"
  Incorrecto: "PAFI mayor a 150" (meta)

**CRITERIOS DE EXCLUSION GLOBAL:**
1. No anotar valores inferidos desde campos estructurados no visibles en el texto.
2. No anotar metas terapeuticas ni rangos objetivo como si fueran mediciones reales.
3. No anotar gases venosos dentro de etiquetas arteriales.
4. No anotar PAS ni PAD (estas etiquetas no existen en el esquema).
5. No anotar fragmentos incompletos si el texto ofrece una version completa mas adecuada.

**CRITERIOS DE ANOTACION:**
1. Incluir siglas, abreviaturas y unidades cuando hagan parte directa de la mencion anotada.
2. Para EDAD, anotar la expresion etaria y no exigir un prefijo explicito.
3. Para DX, anotar la mencion diagnostica sin arrastrar contexto narrativo extra.
4. Buscar TODAS las variantes de cada entidad listadas arriba.
5. Extraer TODAS las entidades que encuentres, priorizando recall sin inventar entidades.
6. OBLIGATORIO: calcular start y end para cada entidad.
   - start: posicion del primer caracter de la entidad en el texto
   - end: posicion del caracter inmediatamente posterior al ultimo caracter de la entidad
   - Los offsets deben quedar alineados exactamente con el texto de entrada

Texto a analizar:
{text}

Extrae las entidades en formato JSON estructurado."""


# =============================================================================
# FUNCIONES DE FORMATEO DE PROMPTS
# =============================================================================


def build_user_prompt(
    text: str,
    entity_types: List[str],
    schema: dict,
    *,
    include_schema: bool = True,
) -> str:
    """Construye el prompt de usuario para extraccion de entidades (zero-shot)."""
    labels = ", ".join(entity_types) if entity_types else "(sin etiquetas)"
    schema_block = ""
    if include_schema:
        schema_json = json.dumps(schema, ensure_ascii=False)
        schema_block = "Esquema JSON estricto:\n" + schema_json + "\n\n"

    base_prompt = EXTRACTION_PROMPT.format(text=text)
    return (
        base_prompt
        + "\n\n"
        + "Etiquetas permitidas: "
        + labels
        + "\n"
        + "Incluye SIEMPRE la clave 'entities'. Si no hay entidades, devuelve []\n"
        + "No incluyas texto explicativo fuera del JSON.\n"
        + schema_block
    )
