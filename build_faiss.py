# build_faiss.py

# -----------------------
# 1. Imports y constantes
# -----------------------

import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Ruta al JSON que generaste con "fetch_full_articles.py".
# Asegurate de que el nombre coincida exactamente con tu archivo.
INPUT_JSON = "articulos_completos.json"

# Carpeta donde tenés los .txt, en caso quieras usar esos en lugar del JSON.
# Si preferís leer los .txt en vez del JSON, seteá USE_JSON = False.
ARTICULOS_TXT_DIR = "articulos_txt"
USE_JSON = True  # Cambiá a False si querés leer desde los .txt en lugar del JSON

# Nombre de salida para el índice FAISS y el mapping
FAISS_INDEX_FILE = "noticias_politica.index"
MAPPING_PICKLE_FILE = "mapping_id2chunk.pkl"


# Chunking parameters
MIN_CHARS = 20           # minimum characters to keep a chunk
MAX_CHARS = 1000         # maximum characters per chunk before splitting
OVERLAP_CHARS = 200      # overlap characters between subchunks


# ---------------------------
# 2. Funciones de carga de datos
# ---------------------------

def cargar_articulos_desde_json(path_json):
    """
    Lee el JSON que contiene la lista de artículos completos.
    Cada elemento de la lista debe ser un diccionario con al menos la clave "texto".
    Retorna esa lista de diccionarios.
    """
    if not os.path.isfile(path_json):
        raise FileNotFoundError(f"No existe el archivo JSON: {path_json}")
    with open(path_json, "r", encoding="utf-8") as f:
        articulos = json.load(f)
    return articulos


def cargar_articulos_desde_txt(dir_txt):
    """
    Recorre la carpeta dir_txt, lee cada archivo .txt y devuelve
    una lista de diccionarios con:
      - "id": el nombre del archivo (sin extensión) como identificador
      - "texto": todo el contenido del .txt
    Asume que todos los archivos dentro de dir_txt terminan en .txt.
    """
    articulos = []
    archivos = sorted(os.listdir(dir_txt))
    for filename in archivos:
        if not filename.lower().endswith(".txt"):
            continue
        ruta = os.path.join(dir_txt, filename)
        with open(ruta, "r", encoding="utf-8") as f:
            texto_completo = f.read()
        # Usamos el nombre sin la extensión como id
        doc_id = os.path.splitext(filename)[0]
        articulos.append({"id": doc_id, "texto": texto_completo})
    return articulos


# -----------------------------------
# 3. Función para chunkear (por párrafos)
# -----------------------------------

def chunkear_por_parrafos(articulos):
    """
    Toma una lista de artículos (cada uno con al menos 'id' y 'texto'),
    los divide en párrafos y devuelve una lista de chunks con:
      - "doc_id": id original del artículo
      - "chunk_id": identificador único del chunk (por ej. "art_001_p1")
      - "texto": el texto de ese párrafo
    Ignora párrafos demasiado cortos (< 50 caracteres).
    """
    documentos = []
    for art in articulos:
        # Cada artículo: esperamos que tenga "id" y "texto"
        doc_id = art.get("id", None)
        if doc_id is None:
            # Si el JSON venía con otro campo, podés adaptar aquí
            # Por simplicidad, si falta "id" usamos un temporal
            doc_id = f"art_{articulos.index(art):03d}"
        texto_completo = art.get("texto", "")
        # Dividimos por doble salto de línea: típico en noticias
        parrafos = texto_completo.split("\n\n")
        for i, parrafo in enumerate(parrafos, start=1):
            parrafo = parrafo.strip()
            if len(parrafo) < 50:
                # Salteamos párrafos muy cortos (por ejemplo, títulos o frases sueltas)
                continue
            chunk_id = f"{doc_id}_p{i}"
            documentos.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "texto": parrafo
            })
    return documentos



# -----------------------------------
# 4. Generación de embeddings y FAISS
# -----------------------------------

def crear_indice_faiss(documents, modelo_name="all-MiniLM-L6-v2"):
    """
    Dada la lista de chunks (cada uno con "chunk_id" y "texto"),
    genera embeddings con SentenceTransformer y construye un índice FAISS.
    Retorna el índice FAISS y la lista de embeddings en el mismo orden que 'documents'.
    """
    # 4.1. Cargar el modelo de embeddings
    print("🔄 Cargando modelo de embeddings...")
    modelo = SentenceTransformer(modelo_name)

    # 4.2. Preparar lista de textos para el modelo
    texts = [doc["texto"] for doc in documents]
    print(f"🔢 Generando embeddings para {len(texts)} chunks...")
    
    # 4.3. Crear los embeddings (esto puede tardar un rato si hay muchos documentos)
    embeddings = modelo.encode(texts, show_progress_bar=True, batch_size=32)
    # embeddings será un array NumPy de forma (N, D), donde N = cantidad de chunks y D = dimensión del embedding

    # 4.4. Crear índice FAISS
    # Elegimos L2 (distancia Euclidiana). Para similitud coseno podés usar IndexFlatIP + normalizar los embeddings.
    dimension = embeddings.shape[1]
    print(f"📏 Dimensión de los embeddings: {dimension}")
    index = faiss.IndexFlatL2(dimension)
    print("➕ Agregando embeddings al índice FAISS...")
    index.add(embeddings)  # ahora el índice contiene N vectores

    return index, embeddings


# -----------------------------------
# 5. Guardar índice y mapping
# -----------------------------------
import pickle  # para serializar la lista de documentos

def guardar_indice_y_mapping(index, documents, index_path, mapping_path):
    """
    Guarda el índice FAISS en index_path y la lista 'documents' en mapping_path (con pickle).
    'documents' es la lista de dicts con 'chunk_id', 'doc_id' y 'texto'.
    """
    # 5.1. Guardar índice FAISS
    print(f"💾 Guardando índice FAISS en '{index_path}'...")
    faiss.write_index(index, index_path)

    # 5.2. Guardar mapping (documentos) en pickle
    print(f"💾 Guardando mapping (lista de chunks) en '{mapping_path}'...")
    with open(mapping_path, "wb") as f:
        pickle.dump(documents, f)

    print("✅ Índice y mapping guardados con éxito.")





def build_faiss_index(use_json=True, ):
    """
    Función principal para construir el índice FAISS a partir de artículos de noticias.
    """
    print("\n🚀 Iniciando proceso de build FAISS para noticias de política...\n")

    # 6.1. Cargar artículos
    if USE_JSON:
        print("🔍 Cargando artículos desde JSON...")
        articulos = cargar_articulos_desde_json(INPUT_JSON)
        # Esperamos que cada 'articulo' sea {"id": "...", "texto": "...", ...}
        # Si el JSON tiene otros campos (ej. "titulo", "link"), no los usamos aquí.
    else:
        print("🔍 Cargando artículos desde carpeta de .txt...")
        articulos = cargar_articulos_desde_txt(ARTICULOS_TXT_DIR)
        # Aquí cada 'articulo' es {"id": "<nombre_sin_ext>", "texto": "<contenido>"}

    print(f"   • Cantidad de artículos cargados: {len(articulos)}")

    # 6.2. Dividir en chunks (párrafos)
    print("✂️ Dividiendo artículos en chunks (párrafos)...")
    documents = chunkear_por_parrafos(articulos)
    print(f"   • Total de chunks generados: {len(documents)}")

    # 6.3. Generar embeddings y crear índice FAISS
    index, embeddings = crear_indice_faiss(documents)

    # 6.4. Guardar índice y mapping
    guardar_indice_y_mapping(index, documents, FAISS_INDEX_FILE, MAPPING_PICKLE_FILE)

    print("\n🎉 ¡Proceso completado! Tenés tu índice FAISS listo para usar. 🎉\n")


if __name__ == "__main__":
    # Ejecutamos la función principal para construir el índice FAISS
    build_faiss_index(USE_JSON)
    