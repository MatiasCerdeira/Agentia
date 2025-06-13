# build_faiss.py

import os
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuración
INPUT_JSON = "articulos_completos.json"
ARTICULOS_TXT_DIR = "articulos_txt"
USE_JSON = True
FAISS_INDEX_FILE = "indice_articulos.index"
MAPPING_PICKLE_FILE = "ids_articulos.pkl"

MIN_CHARS = 300
MAX_CHARS = 1000
OVERLAP_CHARS = 200

# ---------------------------
# 1. Carga de datos
# ---------------------------

def cargar_articulos_desde_json(path_json):
    if not os.path.isfile(path_json):
        raise FileNotFoundError(f"No existe el archivo JSON: {path_json}")
    with open(path_json, "r", encoding="utf-8") as f:
        articulos = json.load(f)
    return articulos

def cargar_articulos_desde_txt(dir_txt):
    articulos = []
    archivos = sorted(os.listdir(dir_txt))
    for filename in archivos:
        if not filename.lower().endswith(".txt"):
            continue
        ruta = os.path.join(dir_txt, filename)
        with open(ruta, "r", encoding="utf-8") as f:
            texto_completo = f.read()
        doc_id = os.path.splitext(filename)[0]
        articulos.append({"id": doc_id, "texto": texto_completo})
    return articulos

# ---------------------------
# 2. Chunking por párrafos
# ---------------------------

def chunkear_por_parrafos(articulos):
    documentos = []
    for art in articulos:
        doc_id = art.get("id", f"art_{articulos.index(art):03d}")
        texto_completo = art.get("texto", "")
        parrafos = texto_completo.split("\n\n")
        for i, parrafo in enumerate(parrafos, start=1):
            parrafo = parrafo.strip()
            if len(parrafo) < 50:
                continue
            chunk_id = f"{doc_id}_p{i}"
            documentos.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "texto": parrafo
            })
    return documentos

# ---------------------------
# 3. Embeddings por artículo
# ---------------------------

def crear_indice_faiss_por_articulo(documentos, modelo_name="all-mpnet-base-v2"):
    print("🔄 Cargando modelo de embeddings...")
    modelo = SentenceTransformer(modelo_name)

    print("🔢 Agrupando chunks por documento...")
    from collections import defaultdict
    agrupados = defaultdict(list)
    for doc in documentos:
        agrupados[doc["doc_id"]].append(doc["texto"])

    textos_articulos = []
    articulo_ids = []
    for doc_id, chunks in agrupados.items():
        texto_completo = " ".join(chunks)
        textos_articulos.append(texto_completo)
        articulo_ids.append(doc_id)

    print(f"🧠 Generando embeddings para {len(textos_articulos)} artículos...")
    embeddings = modelo.encode(textos_articulos, show_progress_bar=True, batch_size=32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings, articulo_ids

# ---------------------------
# 4. Guardado
# ---------------------------

def guardar_indice_articulos(index, articulo_ids, index_path, ids_path):
    print(f"💾 Guardando índice FAISS en '{index_path}'...")
    faiss.write_index(index, index_path)
    print(f"💾 Guardando mapping de IDs en '{ids_path}'...")
    with open(ids_path, "wb") as f:
        pickle.dump(articulo_ids, f)
    print("✅ Guardado completo.")

# ---------------------------
# 5. Ejecución principal
# ---------------------------

def build_faiss_index(use_json=True):
    print("\n🚀 Iniciando proceso de build FAISS para noticias de política...\n")

    if use_json:
        print("🔍 Cargando artículos desde JSON...")
        articulos = cargar_articulos_desde_json(INPUT_JSON)
    else:
        print("🔍 Cargando artículos desde TXT...")
        articulos = cargar_articulos_desde_txt(ARTICULOS_TXT_DIR)

    print(f"   • Artículos cargados: {len(articulos)}")

    print("✂️ Chunking por párrafos...")
    documentos = chunkear_por_parrafos(articulos)
    print(f"   • Total de chunks: {len(documentos)}")

    index, embeddings, articulo_ids = crear_indice_faiss_por_articulo(documentos)
    guardar_indice_articulos(index, articulo_ids, FAISS_INDEX_FILE, MAPPING_PICKLE_FILE)

    print("\n🎉 Proceso finalizado: índice de artículos generado.\n")

if __name__ == "__main__":
    build_faiss_index(USE_JSON)
