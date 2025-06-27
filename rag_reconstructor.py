import json
import pickle
import csv
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm


class RAGReconstructor:
    def __init__(self, json_articulos, clustering_pkl):
        self.json_articulos = json_articulos
        self.clustering_pkl = clustering_pkl
        self.texto_por_id = {}
        self.cluster_por_id = {}
        self.grupos = defaultdict(list)
        self.resultados = []

    def cargar_datos(self):
        print("📥 Cargando artículos originales y resultados de clustering...")
        with open(self.json_articulos, "r", encoding="utf-8") as f:
            articulos = json.load(f)
        with open(self.clustering_pkl, "rb") as f:
            clustering = pickle.load(f)

        self.texto_por_id = {}
        for articulo in articulos:
            for chunk in articulo.get("chunks", []):
                chunk_id = chunk["chunk_id"]
                texto = chunk.get("texto", "")
                self.texto_por_id[chunk_id] = texto
        self.cluster_por_id = {x["id"]: x["cluster"] for x in clustering}

    def agrupar_por_cluster(self):
        print("🗂️ Agrupando artículos por cluster...")
        for art_id, cluster_id in self.cluster_por_id.items():
            self.grupos[cluster_id].append(art_id)


class ImportanciaClassifier:
    def __init__(self):
        print("🧠 Cargando modelo de clasificación de importancia...")
        self.modelo = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

    def clasificar(self, texto: str):
        try:
            pred = self.modelo(texto[:512])[0]
            return pred["label"], round(pred["score"], 3)
        except Exception as e:
            return "Desconocido", 0.0


class RAGPipeline:
    def __init__(self, reconstructor: RAGReconstructor, clasificador: ImportanciaClassifier):
        self.reconstructor = reconstructor
        self.clasificador = clasificador

    def ejecutar(self):
        self.reconstructor.cargar_datos()
        self.reconstructor.agrupar_por_cluster()

        print("🔎 Reconstruyendo textos estilo RAG y clasificando...")
        for cluster_id, ids in tqdm(self.reconstructor.grupos.items()):
            textos = [self.reconstructor.texto_por_id.get(doc_id, "") for doc_id in ids]
            texto_reconstruido = "\n\n".join(textos)
            importancia, confianza = self.clasificador.clasificar(texto_reconstruido)

            self.reconstructor.resultados.append({
                "cluster_id": int(cluster_id),
                "articulos_incluidos": ids,
                "texto_reconstruido": texto_reconstruido,
                "importancia": importancia,
                "confianza": confianza
            })

    def exportar(self, output_json="noticias_reconstruidas_clasificadas.json", output_csv="noticias_reconstruidas_clasificadas.csv"):
        print(f"💾 Guardando resultados en '{output_json}'...")
        with open(output_json, "w", encoding="utf-8") as f_json:
            json.dump(self.reconstructor.resultados, f_json, indent=2, ensure_ascii=False)

        print(f"📄 Guardando resumen en tabla '{output_csv}'...")
        with open(output_csv, "w", newline='', encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["cluster_id", "articulos_incluidos", "confianza"])
            for r in self.reconstructor.resultados:
                writer.writerow([
                    r["cluster_id"],
                    ";".join(map(str, r["articulos_incluidos"])),
                    r["confianza"]
                ])
        print("✅ Proceso finalizado: JSON y CSV generados.")
