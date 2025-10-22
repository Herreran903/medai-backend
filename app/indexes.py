# Este script define una función para garantizar que ciertos índices sean creados en una colección de MongoDB.
# Los índices optimizan las consultas en campos específicos, mejorando el rendimiento de la base de datos.

from pymongo.database import Database


# Función principal para garantizar la creación de índices en la colección "episodes".
# Parámetros:
# - db (Database): Objeto de base de datos de pymongo que permite interactuar con MongoDB.
# No retorna ningún valor, pero asegura que los índices necesarios estén presentes.
def ensure_indexes(db: Database) -> None:
    eps = db.episodes  # Obtiene la colección "episodes" de la base de datos.

    # Crea un índice en el campo "updated_at" para optimizar consultas basadas en la fecha de actualización.
    eps.create_index("updated_at")

    # Crea un índice en el campo anidado "notes.note_date" para mejorar el rendimiento
    # de consultas que filtran o buscan por fechas específicas dentro de las notas.
    eps.create_index("notes.note_date")

    # Crea un índice en el campo "notes.content_hash" para acelerar búsquedas
    # basadas en el hash del contenido de las notas, útil para evitar duplicados o verificar integridad.
    eps.create_index("notes.content_hash")
