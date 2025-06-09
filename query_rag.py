# query_rag.py

# -----------------------
# 1. Imports y constantes
# -----------------------

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Nombres de los archivos que generamos en el paso anterior
FAISS_INDEX_FILE = "noticias_politica.index"
MAPPING_PICKLE_FILE = "mapping_id2chunk.pkl"

# Cantidad de vecinos a recuperar (podés ajustar según tus pruebas)
TOP_K = 10

# Modelo de embeddings (mismo que usamos para construir el índice)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------------
# 2. Funciones de carga: índice + mapping
# -----------------------------------

def cargar_indice(index_path):
    """
    Lee el índice FAISS desde disco.
    """
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"No existe el índice en: {index_path}")
    print(f"🔍 Cargando índice FAISS desde '{index_path}'...")
    index = faiss.read_index(index_path)
    return index

def cargar_mapping(mapping_path):
    """
    Carga la lista de documentos (mapping) desde el pickle.
    Cada elemento es un dict con 'doc_id', 'chunk_id' y 'texto'.
    """
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"No existe el mapping en: {mapping_path}")
    print(f"🔍 Cargando mapping (lista de chunks) desde '{mapping_path}'...")
    with open(mapping_path, "rb") as f:
        documents = pickle.load(f)
    return documents

# -----------------------------------
# 3. Función para hacer query en FAISS
# -----------------------------------

def buscar_chunks_faiss(index, documents, modelo, pregunta, top_k=TOP_K):
    """
    1. Genera el embedding de la 'pregunta'.
    2. Busca los top_k vecinos más cercanos en 'index'.
    3. Devuelve una lista de dicts con:
       - 'chunk_id'
       - 'doc_id'
       - 'texto' (párrafo completo)
       - 'distancia' (valor retornado por FAISS)
    """
    # 3.1. Vectorizar la pregunta
    print(f"\n🔎 Vectorizando la consulta: \"{pregunta}\"")
    embedding_query = modelo.encode([pregunta])  # devuelve shape (1, dim)

    # 3.2. Realizar la búsqueda en FAISS
    # index.search recibe (array_de_queries, k) y devuelve (distancias, índices)
    distancias, indices = index.search(embedding_query, top_k)
    distancias = distancias[0]  # porque solo pasamos 1 consulta
    indices = indices[0]

    resultados = []
    for i, idx in enumerate(indices):
        # Si idx = -1 puede significar que no hay más vectores; pero con IndexFlatL2 
        # normalmente no pasa si top_k < total de vectores
        if idx < 0 or idx >= len(documents):
            continue
        doc = documents[idx]
        resultados.append({
            "chunk_id": doc["chunk_id"],
            "doc_id": doc["doc_id"],
            "texto": doc["texto"],
            "distancia": float(distancias[i])
        })

    return resultados

# -----------------------------------
# 4. Bloque principal para prueba
# -----------------------------------

if __name__ == "__main__":
    print("\n🚀 Iniciando consulta RAG...")

    # 4.1. Cargar índice FAISS
    index = cargar_indice(FAISS_INDEX_FILE)

    # 4.2. Cargar mapping (lista de chunks)
    documents = cargar_mapping(MAPPING_PICKLE_FILE)

    # 4.3. Cargar modelo de embeddings (mismo que en build)
    print("🔄 Cargando modelo de embeddings para query...")
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 4.4. Pedirle al usuario una pregunta (o definirla fijo para la prueba)
    # Podés descomentar la línea con input() si querés un prompt interactivo:
    # pregunta = input("\n✍️  Ingresá tu consulta de política argentina: ")
    # Para probar rápido, definimos una pregunta fija:
    pregunta = "¿Qué dijeron los medios acerca de la Ley de Educación hoy?"

    # 4.5. Buscar los chunks más relevantes
    resultados = buscar_chunks_faiss(index, documents, modelo, pregunta, top_k=TOP_K)

    # 4.6. Mostrar en pantalla los resultados encontrados
    print(f"\n🏅 Top {TOP_K} chunks más cercanos a la consulta:\n")
    for i, res in enumerate(resultados, start=1):
        print(f"{i}. [doc_id: {res['doc_id']}, chunk_id: {res['chunk_id']} ]")
        print(f"   Distancia: {res['distancia']:.4f}")
        print(f"   Texto: {res['texto'][:200]}...")  # muestro los primeros 200 caracteres
        print()