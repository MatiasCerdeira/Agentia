# cluster_hdbscan.py

import faiss
import csv
import pickle
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize



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


def ejecutar_hdbscan(embeddings, min_cluster_size=5, min_samples=None):
    """
    Ejecuta HDBSCAN sobre embeddings normalizados.
    Args:
        embeddings (np.ndarray): Matriz de embeddings (n_samples, n_features)
        min_cluster_size (int): Tamaño mínimo de cluster
        min_samples (int): Muestras mínimas para definir densidad (por defecto = min_cluster_size)

    Returns:
        labels (np.ndarray): Etiquetas de cluster para cada punto (-1 = outlier)
        clusterer (HDBSCAN): Objeto entrenado para inspección o soft clustering
    """
    if min_samples is None:
        min_samples = min_cluster_size

    # Normalizamos los embeddings para simular distancia coseno
    embeddings_norm = normalize(embeddings, norm='l2')

    # Reducimos dimensionalidad con PCA (por ejemplo a 50 dimensiones)
    print("🔻 Reduciendo dimensionalidad con PCA...")
    pca = PCA(n_components=20)
    reduced = pca.fit_transform(embeddings_norm)

    # Clustering con HDBSCAN sobre los datos reducidos
    print("🔄 Ejecutando clustering HDBSCAN sobre datos reducidos con PCA...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric='euclidean', prediction_data=True)
    labels = clusterer.fit_predict(reduced)

    return labels, clusterer


def asignar_clusters(documentos, labels):
    """
    Asigna el número de cluster a cada documento.
    """
    # Si labels viene como una tupla (ej. (labels, probs)), tomamos solo la primera parte
    if isinstance(labels, tuple):
        labels = labels[0]

    # Asegura que sea lista de enteros
    labels = [int(l) for l in labels]

    for i, doc in enumerate(documentos):
        doc["cluster"] = labels[i]
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
    labels, clusterer = ejecutar_hdbscan(embeddings, min_cluster_size)
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
