# query_rag.py

import json
import pickle
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm


def reconstruir_noticias_y_clasificar(
    json_articulos_path="articulos_completos.json",
    clustering_pkl_path="articulos_clusterizados.pkl",
    output_file="noticias_reconstruidas_clasificadas.json",
    modelo_clasificacion_name="cross-encoder/nli-deberta-v3-base",
    truncar_a=512
):
    """
    Reconstruye noticias por cluster y las clasifica según importancia.
    Parámetros:
        json_articulos_path: str - Ruta al archivo JSON con artículos originales.
        clustering_pkl_path: str - Ruta al archivo .pkl con IDs de artículos y su cluster.
        output_file: str - Ruta de salida para guardar el JSON con reconstrucciones.
        modelo_clasificacion_name: str - Modelo HuggingFace para clasificación de importancia.
        truncar_a: int - Cantidad máxima de caracteres usados para clasificar.
    """
    print("📥 Cargando artículos originales y resultados de clustering...")
    with open(json_articulos_path, "r", encoding="utf-8") as f:
        articulos = json.load(f)
    with open(clustering_pkl_path, "rb") as f:
        clustering = pickle.load(f)

    texto_por_id = {a["id"]: a["texto"] for a in articulos}
    cluster_por_id = {x["id"]: x["cluster"] for x in clustering}

    print("🗂️ Agrupando artículos por cluster...")
    grupos = defaultdict(list)
    for art_id, cluster_id in cluster_por_id.items():
        grupos[cluster_id].append(art_id)

    print("🧠 Cargando modelo de clasificación de importancia...")
    modelo_clasificacion = pipeline("text-classification", model=modelo_clasificacion_name)

    print("🔎 Reconstruyendo textos estilo RAG y clasificando...")
    resultados = []

    for cluster_id, ids in tqdm(grupos.items()):
        textos = [texto_por_id.get(doc_id, "") for doc_id in ids]
        texto_reconstruido = "\n\n".join(textos)
        resumen = texto_reconstruido[:truncar_a]
        try:
            pred = modelo_clasificacion(resumen)[0]
            importancia = pred["label"]
            confianza = round(pred["score"], 3)
        except Exception as e:
            importancia = "Desconocido"
            confianza = 0.0
            pred = {"error": str(e)}

        resultados.append({
            "cluster_id": int(cluster_id),
            "articulos_incluidos": ids,
            "texto_reconstruido": texto_reconstruido,
            "importancia": importancia,
            "confianza": confianza
        })

    print(f"💾 Guardando resultados en '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("✅ Proceso finalizado: noticias reconstruidas y clasificadas.")
    return resultados


if __name__ == "__main__":
    reconstruir_noticias_y_clasificar()
