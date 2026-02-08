# MedAI Backend - Documentación de Tesis

Este documento consolida toda la información técnica y académica del backend MedAI para su inclusión en la tesis de grado. El contenido está organizado para facilitar la redacción de las secciones del documento final.

## Tabla de Contenidos

1. [Descripción General del Backend](#1-descripción-general-del-backend)
2. [Arquitectura y Tecnologías](#2-arquitectura-y-tecnologías)
3. [Estrategia Multi-Modelo de Extracción](#3-estrategia-multi-modelo-de-extracción)
4. [Endpoints y Formato de Datos](#4-endpoints-y-formato-de-datos)
5. [Diseño de la Base de Datos](#5-diseño-de-la-base-de-datos)
6. [Procesamiento de Documentos Clínicos](#6-procesamiento-de-documentos-clínicos)
7. [Despliegue y Configuración](#7-despliegue-y-configuración)
8. [Normalización Terminológica](#8-normalización-terminológica)
9. [Taxonomía de Entidades](#9-taxonomía-de-entidades)
10. [Limitaciones y Trabajo Futuro](#10-limitaciones-y-trabajo-futuro)

---

## 1. Descripción General del Backend

### 1.1 Propósito y Enfoque Funcional

El backend MedAI se desarrolló como un servicio web orientado a la extracción de información clínica estructurada a partir de notas médicas en texto libre, con énfasis en casos de ventilación mecánica. Su objetivo es convertir texto no estructurado (por ejemplo, evolución médica, terapias respiratorias, notas de UCI) en una salida estandarizada de entidades clínicas relevantes, de modo que puedan ser consultadas, almacenadas o utilizadas por otros componentes del sistema (p. ej., frontend, analítica o integración futura).

### 1.2 Funciones Principales

En términos funcionales, el backend permite:

1. **Recibir texto clínico directamente o como documento** (PDF/DOCX/TXT)
2. **Ejecutar extracción de entidades con diferentes estrategias/modelos** (multi-modelo)
3. **Entregar resultados en un formato estructurado** (lista de entidades con tipo, texto y offsets)
4. **Persistir resultados asociados a un episodio clínico** y permitir su recuperación posterior
5. **Exponer endpoints de salud y documentación automática** para operación y consumo

### 1.3 Rol en el Sistema MedAI

El backend se concibió como una capa de servicio que operacionaliza modelos de PLN clínico en un entorno reproducible y consumible por API, habilitando extracción, trazabilidad y despliegue local seguro. No es solo "un modelo corriendo", sino una capa de producto que encapsula decisiones clínicas, operativas y de calidad para que la extracción sea usable en un sistema real.

---

## 2. Arquitectura y Tecnologías

### 2.1 Arquitectura por Capas

El backend se implementa como un servicio REST basado en FastAPI, adoptando un enfoque por capas (API → servicios → modelos), con configuración centralizada y dependencias inyectadas.

#### Capas del Sistema

**A) Capa de API (FastAPI)**
- Expone endpoints REST para extracción individual, extracción por lotes y consulta de resultados
- Publica documentación OpenAPI (Swagger/ReDoc) para facilitar integración y pruebas

**B) Capa de Orquestación (Pipeline)**
- Encapsula la lógica de selección de modelo/variante
- Ejecuta la extracción y valida/estandariza el resultado antes de retornarlo
- Permite un paso opcional de normalización terminológica (UMLS), actualmente deshabilitado en la API

**C) Capa de Modelos (Estrategias de Extracción)**
El backend integra tres estrategias:
- **BiLSTM (TensorFlow)**: orientado a inferencia rápida
- **Transformer (PyTorch + Hugging Face)**: orientado a mayor precisión (RoBERTa, fijado para experimentos)
- **LLM (OpenAI)**: orientado a extracción flexible con salida estructurada (GPT, fijado para experimentos)

**D) Capa de Persistencia (MongoDB)**
- Guarda resultados por episodio y nota
- Soporta deduplicación por hash de contenido
- Permite recuperar resultados por identificador (note_id)

### 2.2 Arquitectura de Microservicios

El backend utiliza una arquitectura de microservicios donde cada modelo NER se ejecuta en un contenedor independiente, proporcionando mejor aislamiento, tiempos de inicio más rápidos y capacidades de escalado independiente.

```
                    ┌─────────────────────┐
                    │   API Gateway       │
                    │   Port: 8000        │
                    │   Size: ~200MB      │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐     ┌──────────────┐
│ Transformer   │      │   BiLSTM     │     │     LLM      │
│ Port: 8001    │      │  Port: 8002   │     │  Port: 8003  │
│ Size: ~1.8GB  │      │  Size: ~1.2GB │     │  Size: ~100MB│
│  RoBERTa      │      │  BiLSTM-CRF   │     │    GPT       │
└───────────────┘      └───────────────┘     └──────────────┘
```

### 2.3 Ventajas de la Arquitectura de Microservicios

| Aspecto | Monolito | Microservicios |
|---------|----------|----------------|
| **Tamaño de imagen gateway** | 3.5GB | 200MB (94% reducción) |
| **Startup gateway** | 10-15s | <2s |
| **Dependencias** | Todas juntas | Aisladas por servicio |
| **Escalado** | Todo o nada | Por servicio independiente |
| **Conflictos de versiones** | Posibles | Imposibles |
| **Latencia HTTP** | N/A | +5-10ms adicional |
| **Complejidad operativa** | Baja | Media |

**Beneficios:**
- **Gateway ligero**: Reducción del 94% en tamaño de imagen (3.5GB → 200MB)
- **Inicio rápido**: Gateway listo en <2s vs 10-15s para monolito
- **Dependencias aisladas**: PyTorch, TensorFlow y LLM SDKs en contenedores separados
- **Sin conflictos de versiones**: Cada modelo puede usar versiones diferentes de bibliotecas
- **Escalado granular**: Escalar solo el servicio Transformer si es el más usado
- **Despliegue selectivo**: Actualizar solo un servicio sin tocar los demás

### 2.4 Tecnologías Empleadas

#### Stack Principal
- **Lenguaje**: Python 3.10+
- **Framework API**: FastAPI + Uvicorn (servicio ASGI)
- **Contenerización**: Docker + docker-compose
- **Persistencia**: MongoDB 6

#### Modelos NER
- **Transformers**: Hugging Face Transformers + PyTorch
- **BiLSTM**: TensorFlow/Keras (modelo local)
- **LLM**: SDKs de OpenAI (GPT) (Claude deshabilitado en experimentos)

#### Procesamiento de Documentos
- **PDF**: PyMuPDF (extracción de texto y limpieza)
- **DOCX**: docx2txt
- **TXT**: procesamiento nativo

#### Exposición Segura
- **Dominio**: Cloudflare
- **Túnel seguro**: Cloudflare Tunnel (sin exponer puertos del host)

### 2.5 Componentes Principales

| Componente | Responsabilidad | Propósito Metodológico |
|------------|-----------------|------------------------|
| **API / Router** | Definir contrato de la API y validar entradas | Asegurar consistencia y usabilidad del servicio |
| **Pipeline** | Orquestar inferencia y post-procesamiento | Desacoplar la API de la lógica de extracción |
| **Registro de modelos** | Centralizar acceso a extractores | Habilitar soporte multi-modelo bajo una interfaz uniforme |
| **Extractores** | Ejecutar la inferencia NER (según arquitectura) | Materializar la extracción clínica |
| **Utilidades de texto** | Convertir PDF/DOCX/TXT a texto | Ampliar aplicabilidad a documentos reales |
| **Configuración** | Parametrizar comportamiento por ambiente | Facilitar despliegue y reproducibilidad |
| **Persistencia** | Guardar/recuperar resultados + deduplicación | Continuidad operacional y auditoría |

### 2.6 Patrones de Diseño

| Patrón / Decisión | Aplicación (Alto Nivel) | Aporte para la Tesis |
|-------------------|-------------------------|----------------------|
| **Repository Pattern** | Abstracción de acceso a datos (`EpisodeRepository`) | Separa lógica de negocio de persistencia, facilita testing y cambios de BD |
| **Service Layer** | Capa de servicios (`ExtractionService`) que encapsula lógica de negocio | Centraliza lógica de extracción, mejora mantenibilidad y reutilización |
| **Fábrica (Factory)** | Creación centralizada de la app y middleware | Facilita pruebas y configuración por ambiente |
| **Inyección de dependencias** | Configuración y recursos inyectados a endpoints | Desacopla la capa API de infraestructura |
| **Registro (Service Registry)** | Catálogo de extractores disponibles | Soporta multi-modelo con interfaz uniforme |
| **Estrategia y fachada** | Selección de variante/proveedor sin cambiar contrato | Habilita comparación de modelos y escalamiento iterativo |
| **Caché** | Reutilización de modelos pesados en memoria | Mejora latencia y evita recargas repetidas |
| **Deduplicación por contenido** | Hash del texto para evitar duplicados | Aporta consistencia y trazabilidad operacional |

#### Implementación de Repository Pattern y Service Layer

El backend implementa una **arquitectura por capas limpia** que separa responsabilidades:

**Repository Pattern (`app/repositories/episode_repository.py`)**:
- Abstrae completamente el acceso a MongoDB
- Encapsula operaciones CRUD sobre episodios y notas
- Maneja deduplicación por `content_hash` sin exponer detalles de implementación
- Facilita testing mediante interfaces claras y permite cambiar el motor de BD sin afectar lógica de negocio

**Service Layer (`app/services/extraction_service.py`)**:
- Encapsula toda la lógica de negocio de extracción
- Orquesta llamadas a microservicios NER, normalización y persistencia
- Desacopla los endpoints (routers) de la lógica de extracción y almacenamiento
- Reduce duplicación de código entre endpoints `/extract` y `/extract-batch`

**Beneficios observados**:
- **Reducción de código en routers**: 37% menos líneas en `app/routers/extract.py`
- **Mayor testeabilidad**: Cada capa puede probarse de forma aislada
- **Mantenibilidad mejorada**: Cambios en persistencia o lógica de negocio no afectan la API
- **Reutilización**: La misma lógica sirve para extracción individual y por lotes

---

## 3. Estrategia Multi-Modelo de Extracción

### 3.1 Justificación

Una contribución práctica del backend es integrar varias familias de modelos bajo un mismo servicio, permitiendo seleccionar el extractor según restricciones de latencia, costo, disponibilidad de GPU o necesidad de flexibilidad.

En texto clínico real, existe variabilidad lingüística, abreviaturas y redacción telegráfica. Un enfoque multi-modelo permite balancear calidad vs. costo según el caso de uso.

### 3.2 Modelos Soportados

| Modelo | Enfoque | Ventaja Principal | Limitación Típica | Escenario Recomendado |
|--------|---------|-------------------|-------------------|----------------------|
| **BiLSTM** (BiLSTM-CRF) | Modelo secuencial entrenado en dominio | Alta velocidad y operación offline | Menor flexibilidad semántica | Procesamiento intensivo con baja latencia |
| **Transformer** (RoBERTa) | Token classification (BIO) | Mayor precisión para NER | Mayor costo computacional | Extracción "default" por calidad |
| **LLM** (GPT) | Extracción guiada por esquema | Flexibilidad ante redacción variable | Costo y dependencia de API | Casos complejos o entidades difíciles |

### 3.3 Control de Variantes

El backend soporta variantes mediante el parámetro `model_variant`, aunque actualmente están fijadas para los experimentos:

**Para Transformers:**
- `roberta` (fijado): Modelo RoBERTa en español para experimentos

**Para LLM:**
- `gpt` (fijado): OpenAI GPT para experimentos

El diseño multi-variante facilita comparar modelos y sostener un proceso incremental de mejora sin cambiar el contrato de la API. Para los experimentos de esta tesis, se fijaron las variantes para garantizar reproducibilidad.

### 3.4 Coexistencia de Frameworks

El backend integra:
- **PyTorch** (Transformers)
- **TensorFlow** (BiLSTM)
- **SDKs de proveedores** (LLMs)

Esta coexistencia responde a la realidad del desarrollo: cada modelo fue entrenado/operacionalizado con herramientas distintas, y el backend actúa como capa integradora para exponerlos de forma homogénea.

### 3.5 Resolución y Reutilización de Modelos

Para evitar recargar pesos de modelos en cada solicitud (lo cual aumenta latencia y consumo), el backend adopta un enfoque de reutilización en memoria:

- **Transformers**: Se cachean en memoria para que el costo de carga se pague una sola vez (actualmente fijado a RoBERTa)
- **Registro central de modelos**: Actúa como fuente de verdad de extractores disponibles, reduciendo dispersión de inicialización
- **Carga diferida**: El diseño del backend difiere la carga pesada hasta el primer uso efectivo de extracción (mejor experiencia de arranque del servicio)

**Lectura metodológica**: Esta estrategia reduce tiempo de respuesta en operación sostenida y permite comparar variantes sin reconfigurar el servicio completo.

---

## 4. Endpoints y Formato de Datos

### 4.1 Principios de Diseño del Contrato

1. **Interfaz orientada a flujos reales**: La API acepta entradas en texto o documento, ya que en práctica clínica y en flujos institucionales las notas pueden existir como PDF/DOCX

2. **Metadatos clínicos mínimos para trazabilidad**: Se solicita `episode_id` y `note_date` para organizar resultados por episodio/nota, y facilitar auditoría y recuperación

3. **Respuesta tipo "acuse" con expansión opcional**: Para evitar respuestas excesivamente grandes (texto completo + entidades), la API retorna un acuse con `id` y permite `expand=true` cuando se requiere el resultado completo en la misma respuesta

4. **Documentación automática**: Se habilitan `docs`, `redoc` y `openapi.json`, favoreciendo la reproducibilidad y la comunicación del contrato

### 4.2 Endpoints Principales

| Método | Ruta | Propósito | Salida (Alto Nivel) |
|--------|------|-----------|-------------------|
| `GET` | `/health` | Verificación de vida del servicio | `{"status":"ok"}` |
| `POST` | `/extract` | Extraer entidades desde texto o archivo | Acuse + opcionalmente resultado |
| `POST` | `/extract-batch` | Extraer desde múltiples archivos | Listado de acuses por archivo |
| `GET` | `/notes/{note_id}` | Recuperar resultado almacenado | Texto + entidades + metadatos |

### 4.3 Formato de Request (POST /extract)

**Entrada típica:**
- **Contenido**: `text` (directo) o `file` (PDF/DOCX/TXT)
- **Selección de modelo**:
  - `model`: `lstm` | `transformer` | `llm`
  - `model_variant`: Para transformer: `roberta` (fijado); para llm: `gpt` (fijado)
- **Metadatos operativos**:
  - `episode_id`: Identificador del episodio clínico (requerido; la API rechaza si falta)
  - `note_date`: Fecha clínica de la nota (ISO 8601, requerida; la API rechaza si falta)
- **Comportamiento**:
  - `save`: Guardar o no guardar resultado (default: `true`)
  - `expand`: Incluir o no el resultado completo en la misma respuesta (default: `false`)
  - `normalize`: Activar normalización UMLS (parámetro aceptado pero actualmente ignorado)

### 4.4 Formato de Response

**Salida típica (acuse):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "stored": true,
  "entity_count": 5
}
```

**Salida expandida (expand=true):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "stored": true,
  "entity_count": 5,
  "result": {
    "text": "Paciente con FiO2 60%, PEEP 8 cmH2O",
    "entities": [
      {
        "type": "FIO2",
        "text": "FiO2 60%",
        "start": 13,
        "end": 21,
        "code": "60"
      },
      {
        "type": "PEEP",
        "text": "PEEP 8 cmH2O",
        "start": 23,
        "end": 35,
        "code": "8"
      }
    ],
    "meta": {
      "model": "transformer",
      "inference_time_ms": 1066.54,
      "entity_count": 2
    }
  }
}
```

### 4.5 Estructura de Entidades

Las entidades incluyen offsets (start/end), lo que permite trazabilidad y visualización (resaltar entidades en el texto) sin ambigüedad.

| Campo | Descripción | Valor Metodológico |
|-------|-------------|-------------------|
| `type` | Etiqueta de entidad (p. ej., `FIO2`, `PEEP`, `DX`) | Habilita agregación y análisis por categoría |
| `text` | Fragmento exacto extraído | Verificabilidad clínica y auditoría |
| `start` / `end` | Offsets del fragmento en el texto | Trazabilidad y renderizado en interfaces |
| `code` | Valor normalizado extraído (número, texto o relación) | Facilita análisis cuantitativo y comparaciones |

**Nota**: El uso de offsets permite "anclar" cada entidad al texto fuente sin almacenar reglas o interpretaciones externas.

### 4.6 Flujo de Procesamiento

```
Cliente
  │  (text o file + metadatos + selección de modelo)
  ▼
API (/extract o /extract-batch)
  │ 1) Validación de campos mínimos
  │ 2) Si hay archivo: documento→texto (PDF/DOCX/TXT)
  ▼
Pipeline
  │ 3) Resolución de extractor (modelo + variante)
  │ 4) Inferencia NER → lista de entidades
  │ 5) Validación/estandarización de salida
  │ 6) (Opcional, actualmente deshabilitado) Normalización terminológica (UMLS)
  ▼
Persistencia (opcional)
  │ 7) Guardado por episodio + deduplicación por hash
  ▼
Respuesta
  │ 8) Acuse (id, stored, entity_count) + expand opcional
  ▼
Cliente
```

---

## 5. Diseño de la Base de Datos

### 5.1 Estrategia de Almacenamiento

**Colección principal**: `episodes`

**Estrategia**:
- Un documento MongoDB representa un **EPISODIO clínico** (agrupador lógico), cuyo identificador es provisto por el cliente (`episode_id`) y se usa como clave primaria del documento
- Dentro de cada episodio se almacena un arreglo embebido `notes` que contiene las notas procesadas (cada nota incluye texto original, entidades extraídas y metadatos)

### 5.2 Estructura del Documento

```json
{
  "_id": "<episode_id>",
  "created_at": { "$date": "2026-02-01T00:00:00Z" },
  "updated_at": { "$date": "2026-02-01T00:00:00Z" },
  "notes": [
    {
      "note_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "nota_001.pdf",
      "source_system": "api.extract",
      "text": "<texto_original_extraido_o_enviado>",
      "entities": [
        {
          "type": "FIO2",
          "text": "FiO2 60%",
          "code": "60",
          "start": 14,
          "end": 22
        }
      ],
      "meta": {
        "model": "transformer",
        "inference_time_ms": 1066.54,
        "entity_count": 1
      },
      "model": "transformer",
      "note_date": "2024-01-15T10:30:00+00:00",
      "created_at": { "$date": "2026-02-01T00:00:00Z" },
      "content_hash": "<sha256_hex_del_texto>"
    }
  ]
}
```

**Nota**: En `entities`, el campo `code` representa el valor normalizado derivado del texto (no códigos terminológicos en la salida actual).

### 5.3 Metadatos Almacenados

#### Nivel de Episodio (documento raíz)
- `_id`: episode_id (string, provisto por el cliente)
- `created_at`: Fecha de creación del episodio en el sistema (BSON Date)
- `updated_at`: Fecha de última actualización del episodio (BSON Date)

#### Nivel de Nota (objeto dentro de notes[])
- `note_id`: UUID string generado por el backend
- `filename`: Nombre original del archivo (si aplica; puede ser null)
- `source_system`: String que identifica el origen ("api.extract" o "api.extract-batch")
- `note_date`: String ISO 8601 de fecha clínica de la nota (requerida por la API para guardar)
- `created_at`: Fecha de procesamiento/guardado (BSON Date)
- `model`: String del modelo solicitado en la API ("lstm" | "transformer" | "llm")
- `content_hash`: SHA-256 del texto, usado para deduplicación por contenido dentro del mismo episodio

#### Metadatos del Proceso de Extracción (notes.meta)
- `model`: Identificador del modelo utilizado (e.g., "lstm", "transformer", "llm")
- `inference_time_ms`: Tiempo de inferencia en milisegundos
- `entity_count`: Número de entidades extraídas

### 5.4 Deduplicación e Índices

#### Deduplicación (por contenido)
- Se calcula `content_hash = SHA-256(texto)`
- Si ya existe una nota con el mismo `content_hash` dentro del mismo episodio, el backend NO inserta una nueva nota; retorna el `note_id` existente

#### Índices Creados
- `updated_at`: Ordenamiento/consulta por última actualización del episodio
- `notes.note_date`: Consulta temporal por fecha clínica de la nota
- `notes.content_hash`: Deduplicación eficiente por hash

**Lectura técnica**: Estos índices soportan (i) ordenamiento/consulta por última actualización del episodio, (ii) consulta temporal por fecha clínica de la nota, y (iii) deduplicación eficiente por hash.

### 5.5 Texto Académico para Tesis

"El backend almacena los resultados de extracción en MongoDB utilizando una colección principal denominada 'episodes'. Cada documento representa un episodio clínico identificado por un _id (episode_id) y contiene un arreglo embebido 'notes' con las notas procesadas. Cada nota incluye el texto original, la lista de entidades extraídas (con tipo, span textual, valor normalizado y offsets start/end), metadatos del proceso (modelo utilizado, tiempo de inferencia en milisegundos y conteo de entidades) y atributos de trazabilidad como note_id y content_hash. Adicionalmente, se emplean índices sobre updated_at, notes.note_date y notes.content_hash para optimizar consultas temporales y soportar deduplicación por contenido, garantizando trazabilidad y consistencia en el almacenamiento de resultados."

---

## 6. Procesamiento de Documentos Clínicos

### 6.1 Justificación

En sistemas clínicos reales, el texto puede venir embebido en documentos; por ello se incorporó conversión documento→texto para ampliar la aplicabilidad del sistema.

### 6.2 Formatos Soportados

| Formato | Razón de Soporte | Riesgos/Limitaciones |
|---------|------------------|---------------------|
| **TXT** | Interoperabilidad directa | Variabilidad de encoding |
| **PDF** | Frecuente en exportaciones institucionales | Orden de lectura y saltos de línea |
| **DOCX** | Notas y reportes editables | Dependencia de extracción por librería |

### 6.3 Limpieza y Normalización del Texto

El backend aplica limpieza moderada, especialmente para PDFs:

- Normalización de saltos de línea
- Reconstrucción de palabras cortadas por guiones
- Reducción de espacios múltiples
- Preservación de estructura por párrafos

**Metodológicamente**, esta limpieza busca mejorar la calidad de entrada para NER sin transformar la semántica clínica.

### 6.4 Tecnologías de Procesamiento

- **PDF**: PyMuPDF (fitz) para extracción de texto y limpieza
- **DOCX**: docx2txt para conversión directa a texto plano
- **TXT**: Procesamiento nativo con manejo de encodings (UTF-8, latin-1, etc.)

---

## 7. Despliegue y Configuración

### 7.1 Justificación del Despliegue Local

El sistema integra múltiples modelos de IA. En escenarios donde se ejecutan:
- Un modelo BiLSTM
- Un Transformer
- Un LLM (o su integración)

Los costos de infraestructura cloud para inferencia (CPU/RAM/GPU, almacenamiento, egress) pueden ser elevados. En consecuencia, se adoptó un despliegue **local (en máquina del autor)** para:

- Controlar costos operativos
- Acelerar iteración experimental
- Reducir exposición de infraestructura con datos sensibles

### 7.2 Contenerización con Docker

El backend se empaquetó en contenedores (Docker), y se orquestó localmente con `docker-compose`.

**Metodológicamente**, esta elección aporta:
- Reproducibilidad del entorno
- Aislamiento de dependencias (ML stack)
- Facilidad de arranque y mantenimiento

Se disponen variantes para:
- **Desarrollo** (`docker-compose.dev.yml`): con recarga automática
- **Producción local** (`docker-compose.prod.yml`): sin recarga, optimizado para estabilidad

### 7.3 Configuración por Ambiente

El backend separa configuración del código para favorecer reproducibilidad:

- Configuración por variables de entorno
- Archivos `.env.dev` y `.env.prod` para desarrollo/producción local
- Parámetros para modelos (IDs, claves, orígenes CORS)

#### Grupos de Configuración

| Grupo | Ejemplos de Parámetros | Finalidad |
|-------|------------------------|-----------|
| **API** | `HOST`, `PORT`, `LOG_LEVEL` | Control de ejecución y observabilidad |
| **CORS** | `CORS_ORIGINS` | Habilitar frontend confiable |
| **Persistencia** | `MONGODB_URI`, `MONGODB_DB` | Conexión a almacenamiento |
| **Modelos** | IDs de modelos Transformer, rutas locales BiLSTM | Selección de pesos/versiones |
| **LLM** | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Habilitar proveedor LLM (si se usa) |
| **UMLS** | `UMLS_APIKEY` | Habilitar normalización terminológica |

### 7.4 Dominio y Túnel Seguro con Cloudflare

Para permitir acceso remoto sin exponer puertos del host, se utilizó Cloudflare:

- **Dominio**: Adquirido en Cloudflare
- **Publicación**: Mediante Cloudflare Tunnel

**Ventajas metodológicas**:
- Terminación TLS/HTTPS gestionada por Cloudflare
- Reducción de superficie de ataque (no se abre el servicio directamente al público)
- Control operativo del acceso
- Protección DDoS integrada

**Comando de exposición**:
```bash
cloudflared tunnel run medai-backend
```

### 7.5 Comparación Monolito vs Microservicios

La arquitectura evolucionó de un monolito a microservicios para abordar problemas de tamaño de imagen, tiempos de inicio y gestión de dependencias.

| Aspecto | Monolito | Microservicios | Mejora |
|---------|----------|----------------|--------|
| **Tamaño imagen principal** | 3.5 GB | 200 MB | 94% reducción |
| **Startup gateway** | 10-15s | <2s | 7x más rápido |
| **Aislamiento de dependencias** | No | Sí | Mejor mantenibilidad |
| **Escalado** | Todo junto | Por servicio | Mayor flexibilidad |
| **Conflictos de versiones** | Posibles | Imposibles | Mayor estabilidad |

---

## 8. Normalización Terminológica

### 8.1 Propósito Académico

El backend incluye un módulo de normalización para enlazar entidades (principalmente diagnósticos) con terminologías estándar (p. ej., SNOMED-CT e ICD-10) mediante UMLS.

La normalización agrega valor porque:

- **Facilita interoperabilidad** con sistemas clínicos que requieren códigos
- **Habilita análisis cuantitativo consistente** (mismo concepto → mismo código)
- **Mejora la comparabilidad** entre notas de distinta redacción

### 8.2 Funcionamiento

Si se habilita la normalización (`normalize=true`), el flujo sería:

1. El backend identifica entidades de tipo diagnóstico (`DX`)
2. Envía el texto de la entidad a la API de UMLS
3. Recibe códigos candidatos de sistemas especificados (SNOMED-CT, ICD-10)
4. Agrega códigos candidatos a la entidad (módulo de normalización; no expuesto por la API actual)

### 8.3 Estado Operativo en el Servicio

Actualmente, la normalización se contempla como un paso opcional del pipeline; sin embargo:

- Se encuentra **temporalmente deshabilitada** en el flujo de la API (decisión orientada a reducir cargas y dependencias adicionales en ejecución)
- El código está implementado y funcional
- El parámetro `normalize=true` es aceptado pero actualmente **no activa** el flujo en la API

### 8.4 Requisitos

- Clave API de UMLS (`UMLS_APIKEY`)
- Registro en https://uts.nlm.nih.gov/
- Especificar sistemas de terminología mediante `systems_csv` (ej: "SNOMEDCT_US,ICD10CM")

---

## 9. Taxonomía de Entidades

### 9.1 Enfoque Clínico

El backend se centra en entidades clínicas especialmente relevantes para ventilación mecánica y cuidado crítico. La taxonomía se organiza por categorías funcionales.

### 9.2 Categorías de Entidades

#### Configuración de Ventilación

| Tipo | Descripción | Intención Clínica |
|------|-------------|-------------------|
| `MODO` | Modo ventilatorio | Tipo de ventilación mecánica |
| `FIO2` | Fracción inspirada de oxígeno | Configuración de oxigenación |
| `PEEP` | Presión positiva al final de la espiración | Soporte respiratorio |
| `FR` | Frecuencia respiratoria | Configuración de ritmo ventilatorio |
| `VT` | Volumen tidal | Volumen entregado por respiración |
| `FLUJO` | Flujo inspiratorio | Velocidad de entrega de gases |
| `I_E` | Relación inspiración/espiración | Tiempo inspiratorio vs espiratorio |
| `SENS` | Sensibilidad del trigger | Esfuerzo requerido para iniciar respiración |

#### Respuesta a Ventilación / Mecánica

| Tipo | Descripción | Intención Clínica |
|------|-------------|-------------------|
| `SAO2` | Saturación arterial de oxígeno | Monitorizar oxigenación |
| `PP` | Presión plateau | Presión alveolar al final de inspiración |
| `PMES` | Presión meseta | Presión en vía aérea |
| `PM` | Presión media | Presión media en vía aérea |

#### Signos Vitales

| Tipo | Descripción | Intención Clínica |
|------|-------------|-------------------|
| `TEMP` | Temperatura corporal | Estado febril |
| `PA` | Presión arterial | Hemodinamia general |
| `PAS` | Presión arterial sistólica | Máxima presión arterial |
| `PAD` | Presión arterial diastólica | Mínima presión arterial |
| `PAM` | Presión arterial media | Presión de perfusión |
| `FC` | Frecuencia cardíaca | Estado cardiovascular |
| `GLICEMIA` | Glucemia | Control metabólico |
| `POSTURA` | Posición del paciente | Posición terapéutica (prono, supino) |

#### Antropometría

| Tipo | Descripción | Intención Clínica |
|------|-------------|-------------------|
| `EDAD` | Edad del paciente | Contexto demográfico |
| `PESO` | Peso | Cálculo de dosis y volumen tidal |
| `TALLA` | Estatura | Cálculo de volumen tidal ideal |

#### Gases Arteriales

| Tipo | Descripción | Intención Clínica |
|------|-------------|-------------------|
| `PH` | pH arterial | Estado ácido-base |
| `PACO2` | Presión parcial de CO2 | Ventilación alveolar |
| `HCO3` | Bicarbonato | Componente metabólico |
| `BE` | Exceso de base | Alteración ácido-base |
| `PAO2` | Presión parcial de O2 | Oxigenación arterial |
| `PAFI` | Relación PaO2/FiO2 | Severidad de insuficiencia respiratoria |

#### Observaciones Clínicas

| Tipo | Descripción | Intención Clínica |
|------|-------------|-------------------|
| `DX` | Diagnóstico | Impresión clínica o diagnóstico formal |

### 9.3 Total de Tipos de Entidad

El sistema reconoce **28 tipos de entidades clínicas** organizadas en 6 categorías funcionales, todas específicas al dominio de ventilación mecánica y cuidado crítico.

---

## 10. Limitaciones y Trabajo Futuro

### 10.1 Limitaciones Actuales

| Tema | Situación Actual | Impacto |
|------|------------------|---------|
| **Normalización UMLS** | Presente como capacidad; desactivada en API | Limita interoperabilidad con sistemas externos |
| **Autenticación/autoría** | No implementada | No hay control de acceso por usuario |
| **Manejo de cargas altas** | Multi-modelo en un servicio | Posible cuello de botella en alta demanda |
| **Textos muy largos** | Truncamiento en Transformers | Pérdida de información en notas extensas |
| **Observabilidad avanzada** | Solo health checks básicos | Dificulta diagnóstico de problemas |
| **Portabilidad a cloud** | Despliegue local por costo | Limita escalabilidad horizontal |

### 10.2 Trabajo Futuro Sugerido

#### Corto Plazo
1. **Activar normalización UMLS** con control de costo/latencia y cachés
2. **Implementar autenticación básica** (tokens, API keys)
3. **Agregar métricas de rendimiento** (latencia por modelo, throughput)

#### Mediano Plazo
4. **Implementar segmentación de textos largos** por ventanas + agregación
5. **Agregar circuit breaker** para resiliencia ante fallos de servicios
6. **Configurar log aggregation** (ELK/Loki) para debugging centralizado

#### Largo Plazo
7. **Migrar a Kubernetes** si se requiere mayor escala y orquestación
8. **Implementar rate limiting** para control de uso
9. **Desarrollar modo híbrido** (cloud para frontend, local para inferencia)
10. **Agregar tracing distribuido** (Jaeger/Zipkin) para observabilidad completa

### 10.3 Consideraciones de Escalabilidad

**Escalado Vertical (Actual)**:
- Máquina local con recursos limitados
- Adecuado para demostración y desarrollo

**Escalado Horizontal (Futuro)**:
- Replicar servicios NER independientemente
- Load balancing entre réplicas
- Requiere orquestador (Docker Swarm o Kubernetes)

### 10.4 Mejoras en Modelos

1. **Fine-tuning continuo**: Incorporar feedback de usuarios para mejorar modelos
2. **Ensemble de modelos**: Combinar predicciones de múltiples modelos
3. **Modelos especializados**: Entrenar modelos específicos por tipo de entidad
4. **Validación cruzada**: Comparar resultados entre modelos para detectar inconsistencias

---

## Apéndices

### Apéndice A: Comandos Útiles de Operación

#### Gestión de Servicios
```bash
# Iniciar todos los servicios
docker-compose -f docker-compose.dev.yml up --build

# Ver logs de todos los servicios
docker-compose -f docker-compose.dev.yml logs -f

# Ver logs de un servicio específico
docker-compose -f docker-compose.dev.yml logs -f gateway

# Detener servicios
docker-compose -f docker-compose.dev.yml down

# Detener y eliminar volúmenes
docker-compose -f docker-compose.dev.yml down -v
```

#### Verificación de Estado
```bash
# Health check del gateway
curl http://localhost:8000/health

# Readiness de servicios NER
curl http://localhost:8001/readyz  # Transformer
curl http://localhost:8002/readyz  # BiLSTM
curl http://localhost:8003/readyz  # LLM
```

#### Pruebas
```bash
# Prueba rápida (2-3 minutos)
./quick_test.sh

# Suite completa (5-10 minutos)
./test_microservices.sh
```

### Apéndice B: Estructura de Directorios del Proyecto

```
medai-backend/
├── app/                           # Aplicación principal (Gateway)
│   ├── config.py                  # Configuración centralizada
│   ├── deps.py                    # Dependencias de FastAPI
│   ├── indexes.py                 # Definiciones de índices MongoDB
│   ├── main.py                    # Punto de entrada FastAPI
│   ├── schemas.py                 # Modelos Pydantic
│   ├── repositories/
│   │   └── episode_repository.py  # Repository Pattern para episodios
│   ├── routers/
│   │   └── extract.py             # Endpoints de extracción
│   └── services/
│       ├── extraction_service.py  # Service Layer para extracción
│       ├── ner_client.py          # Cliente para microservicios NER
│       ├── normalizer.py          # Normalización UMLS
│       ├── pipeline.py            # Orquestación de extracción
│       ├── registry.py            # Registro de modelos
│       ├── text_utils.py          # Utilidades de texto
│       ├── translator.py          # Traducción ES-EN
│       └── utils.py               # Utilidades compartidas
├── services/                      # Microservicios NER
│   ├── ner-transformer/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   ├── ner-bilstm/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   └── models/
│   └── ner-llm/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── main.py
├── shared/                        # Código compartido
│   ├── schemas.py                 # Modelos Pydantic compartidos
│   └── utils.py                   # Utilidades compartidas
├── scripts/
│   └── export_openapi.py          # Exportar esquema OpenAPI
├── docker-compose.dev.yml         # Configuración desarrollo
├── docker-compose.prod.yml        # Configuración producción
├── Dockerfile.gateway             # Imagen del gateway
├── requirements.txt               # Dependencias del gateway
├── .env.dev                       # Variables de entorno desarrollo
├── .env.prod                      # Variables de entorno producción
├── quick_test.sh                  # Script de prueba rápida
├── test_microservices.sh          # Suite de pruebas completa
└── README.md                      # Documentación principal
```

### Apéndice C: Glosario de Términos

| Término | Definición |
|---------|------------|
| **NER** | Named Entity Recognition - Reconocimiento de Entidades Nombradas |
| **LSTM** | Long Short-Term Memory - Tipo de red neuronal recurrente |
| **Transformer** | Arquitectura de red neuronal basada en mecanismos de atención |
| **LLM** | Large Language Model - Modelo de Lenguaje de Gran Escala |
| **BETO** | BERT en español - Modelo Transformer pre-entrenado (legacy, no usado) |
| **RoBERTa** | Robustly Optimized BERT - Variante mejorada de BERT |
| **BiLSTM-CRF** | Bidirectional LSTM with Conditional Random Fields |
| **BIO** | Begin-Inside-Outside - Esquema de etiquetado para NER |
| **UMLS** | Unified Medical Language System - Sistema Unificado de Lenguaje Médico |
| **SNOMED-CT** | Systematized Nomenclature of Medicine - Clinical Terms |
| **ICD-10** | International Classification of Diseases, 10th revision |
| **Offset** | Posición de inicio/fin de un fragmento de texto |
| **Episode** | Episodio clínico - Agrupador lógico de notas de un paciente |
| **Content Hash** | Hash SHA-256 del texto para deduplicación |

---

**Nota Final**: Este documento está diseñado para servir como referencia completa para la redacción de la tesis. Los fragmentos aquí incluidos pueden ser adaptados al estilo académico específico requerido por la institución, manteniendo el contenido técnico y metodológico intacto.
