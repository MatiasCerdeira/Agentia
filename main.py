import fetch_links
import fetch_full_articles
import build_faiss
import query_rag
import os
import json

# Lista de RSS “Política” de medios argentinos
RSS_FEEDS = [
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml",
    "https://www.clarin.com/rss/politica/",
    "https://www.perfil.com/feed/politica",
    "https://www.pagina12.com.ar/rss/secciones/economia/notas",
    "https://www.pagina12.com.ar/rss/secciones/sociedad/notas",
    "https://www.pagina12.com.ar/rss/secciones/el-pais/notas",
]


if __name__ == "__main__":

    # 1. Descargar los links de los RSS
    links = fetch_links.fetch_all_feeds(RSS_FEEDS)
    # Guardamos mínima info en JSON -> "links.json"
    with open("links.json", "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2, ensure_ascii=False)
    print(f"✅ Guardé todo en links.json")

    # 2. Descargar los artículos completos
    print("🔄 Descargando artículos completos...")
    fetch_full_articles.download_full_articles(links, "articulos_txt")    

    # 3. Construir el índice FAISS
    print("🔄 Construyendo índice FAISS...")
    build_faiss.build_faiss_index(True)

    # 4. Realizar una consulta RAG
    print("🔄 Realizando consulta RAG...")
    query_rag.run_query("¿Qué es lo más relevante en la política Argentina hoy?")


