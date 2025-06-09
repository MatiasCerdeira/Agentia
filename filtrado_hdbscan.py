# cluster_hdbscan.py

import faiss
import csv
import pickle
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def cargar_embeddings_y_chunks(faiss_index_file, mapping_pickle_file):
    """
    Carga los embeddings desde el índice FAISS y los metadatos de los chunks.
    """
    print("📥 Cargando índice FAISS y mapping...")
    index = faiss.read_index(faiss_index_file)
    embeddings = index.reconstruct_n(0, index.ntotal)

    with open(mapping_pickle_file, "rb") as f:
        documentos = pickle.load(f)

    return embeddings, documentos


def ejecutar_hdbscan(embeddings, min_cluster_size=5):
    """
    Ejecuta HDBSCAN sobre los embeddings.
    """
    print("🔄 Ejecutando clustering HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    return labels


def asignar_clusters(documentos, labels):
    """
    Asocia cada etiqueta de cluster a su correspondiente chunk.
    """
    for i, doc in enumerate(documentos):
        doc["cluster"] = int(labels[i])
    return documentos


def guardar_clusters(documentos, output_path="documentos_clusterizados_hdbscan.pkl"):
    """
    Guarda los documentos con cluster asignado.
    """
    with open(output_path, "wb") as f:
        pickle.dump(documentos, f)
    print(f"✅ Clusters guardados en '{output_path}'")


def mostrar_resumen(documentos):
    """
    Muestra un resumen de cuántos chunks hay en cada cluster.
    """
    print("\n📊 Resumen de clusters:")
    labels = [doc["cluster"] for doc in documentos]
    for c in sorted(set(labels)):
        grupo = [d for d in documentos if d["cluster"] == c]
        nombre = "Outliers" if c == -1 else f"Cluster {c}"
        muestra = grupo[0]["texto"][:200].replace("\n", " ") + "..." if grupo else ""
        print(f"🔹 {nombre}: {len(grupo)} chunks")
        if grupo:
            print("   Ejemplo:", muestra)


def visualizar_pca(embeddings, labels):
    """
    Muestra un gráfico 2D usando PCA coloreado por cluster.
    """
    print("📈 Visualizando clusters con PCA...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.title("Clustering de noticias con HDBSCAN (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()


def clusterizar_noticias_con_hdbscan(
    faiss_index_file="noticias_politica.index",
    mapping_pickle_file="mapping_id2chunk.pkl",
    output_file="documentos_clusterizados_hdbscan.pkl",
    min_cluster_size=5,
    visualizar=True
):
    """
    Función principal para clusterizar noticias con HDBSCAN.
    """
    embeddings, documentos = cargar_embeddings_y_chunks(faiss_index_file, mapping_pickle_file)
    labels = ejecutar_hdbscan(embeddings, min_cluster_size)
    documentos = asignar_clusters(documentos, labels)
    mostrar_resumen(documentos)
    guardar_clusters(documentos, output_file)
    if visualizar:
        visualizar_pca(embeddings, labels)
    with open("documentos_clusterizados_hdbscan.pkl", "rb") as f:
        documentos = pickle.load(f)

    with open("clusters_noticias.csv", "w", newline='', encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["cluster", "doc_id", "chunk_id", "texto"])

        for doc in documentos:
            writer.writerow([doc["cluster"], doc["doc_id"], doc["chunk_id"], doc["texto"][:500]])
            
    print("✅ Exportado como clusters_noticias.csv")
    return documentos  # por si querés usarlos desde otro módulo


if __name__ == "__main__":
    # Si ejecutás este archivo directamente
    clusterizar_noticias_con_hdbscan()
