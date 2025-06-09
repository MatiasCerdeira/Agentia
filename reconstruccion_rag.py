# reconstruccion_rag.py

import os
import pickle
import faiss
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm

# === CONFIG ===
FAISS_INDEX_FILE = "noticias_politica.index"
MAPPING_PICKLE_FILE = "mapping_id2chunk.pkl"
CLUSTERED_PICKLE_FILE = "documentos_clusterizados_hdbscan.pkl"
OUTPUT_JSON = "noticias_generadas_rag_por_cluster.json"
TOP_K = 10
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)
modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === UTILIDADES ===

def construir_faiss_para_cluster(chunks):
    textos = [doc["texto"] for doc in chunks]
    embeddings = modelo.encode(textos, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, textos, embeddings

def consulta_rag(index, textos, query, top_k=TOP_K):
    query_embedding = modelo.encode([query])
    distancias, indices = index.search(query_embedding, top_k)
    chunks_recuperados = [textos[i] for i in indices[0]]
    return chunks_recuperados

def generar_noticia_con_openai(chunks, cluster_id):
    context = "\n\n".join(chunks)
    prompt = f"""Estos son fragmentos de texto extraídos de artículos del cluster {cluster_id}, agrupados automáticamente por similitud semántica.

Tu tarea es reconstruir una noticia completa, coherente y redactada con estilo periodístico, integrando el contenido de estos fragmentos:

---
{context}
---

Redactá la noticia:"""

    try:
        respuesta = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error al consultar OpenAI: {e}")
        return None

# === SCRIPT PRINCIPAL ===

def main():
    with open(CLUSTERED_PICKLE_FILE, "rb") as f:
        documentos = pickle.load(f)

    clusters = defaultdict(list)
    for doc in documentos:
        if doc["cluster"] != -1:
            clusters[doc["cluster"]].append(doc)

    print(f"🔍 Procesando {len(clusters)} clusters con RAG + OpenAI...")

    resultados = {}

    for cluster_id in tqdm(sorted(clusters.keys())):
        chunks = clusters[cluster_id]
        if len(chunks) < 3:
            continue  # ignoramos clusters demasiado chicos

        index, textos, embeddings = construir_faiss_para_cluster(chunks)
        query = f"Reconstruí una noticia a partir de los fragmentos agrupados en el cluster {cluster_id}."
        chunks_mas_relevantes = consulta_rag(index, textos, query, top_k=TOP_K)
        noticia = generar_noticia_con_openai(chunks_mas_relevantes, cluster_id)

        resultados[cluster_id] = {
            "noticia_generada": noticia,
            "cantidad_de_chunks": len(chunks)
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"✅ Reconstrucción completada. Guardado en '{OUTPUT_JSON}'")

if __name__ == "__main__":
    main()
