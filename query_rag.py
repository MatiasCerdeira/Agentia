# reconstruir_noticias_rag.py

import pickle
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm

# Cargar textos reales de los artículos
with open("articulos_completos.json", "r", encoding="utf-8") as f:
    articulos = json.load(f)

# Crear diccionario para lookup rápido por ID
texto_por_id = {art["id"]: art["texto"] for art in articulos}
# Archivos requeridos
FAISS_INDEX_FILE = "indice_articulos.index"
MAPPING_PICKLE_FILE = "ids_articulos.pkl"
OUTPUT_FILE = "noticias_reconstruidas_clasificadas.json"

# Paso 1: Cargar índice FAISS e IDs
print("📥 Cargando índice FAISS e IDs de artículos...")
index = faiss.read_index(FAISS_INDEX_FILE)

with open(MAPPING_PICKLE_FILE, "rb") as f:
    articulo_ids = pickle.load(f)

# Paso 2: Cargar modelo de embeddings
print("🔄 Cargando modelo de embeddings...")
modelo_emb = SentenceTransformer("all-mpnet-base-v2")

# Paso 3: Cargar modelo para clasificación de importancia
print("🧠 Cargando modelo de clasificación de importancia...")
modelo_clasificacion = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

# Paso 4: Recuperar y reconstruir noticias agrupadas
print("🔎 Realizando recuperación y reconstrucción estilo RAG...")
reconstruidas = []

for i, doc_id in enumerate(tqdm(articulo_ids)):
    # Obtener embedding del artículo
    vector = index.reconstruct(i).reshape(1, -1)

    # Buscar los 5 artículos más similares (incluido él mismo)
    D, I = index.search(vector, k=5)

    # Evitar duplicados, unir los textos más cercanos
    ids_similares = list(set([articulo_ids[j] for j in I[0]]))

    textos = [texto_por_id.get(doc_id, "") for doc_id in ids_similares]
    texto_completo = "\n\n".join(textos)
    reconstruidas.append({
        "grupo_id": i,
        "ids_incluidos": ids_similares,
        "texto_reconstruido": texto_completo
    })

# Paso 5: Clasificar importancia
print("📊 Clasificando importancia de los grupos...")
for grupo in tqdm(reconstruidas):
    resumen = grupo["texto_reconstruido"][:512]
    try:
        pred = modelo_clasificacion(resumen)[0]
        grupo["importancia"] = pred["label"]
        grupo["confianza"] = round(pred["score"], 3)
    except Exception as e:
        grupo["importancia"] = "Desconocido"
        grupo["confianza"] = 0.0
        grupo["error"] = str(e)

# Paso 6: Guardar
import json
print(f"💾 Guardando resultados en {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(reconstruidas, f, indent=2, ensure_ascii=False)

print("✅ Proceso completado. Noticias reconstruidas y clasificadas.")