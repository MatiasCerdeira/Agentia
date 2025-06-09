# fetch_rss_sites.py
import feedparser
import json
from datetime import datetime

# Lista de RSS “Política” de medios argentinos
RSS_FEEDS = [
    "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml",
    "https://www.clarin.com/rss/politica/",
    "https://www.perfil.com/feed/politica",
    "https://www.pagina12.com.ar/rss/secciones/economia/notas",
    "https://www.pagina12.com.ar/rss/secciones/sociedad/notas",
    "https://www.pagina12.com.ar/rss/secciones/el-pais/notas",
]

def fetch_all_feeds(feeds):
    """
    Recorre cada RSS y devuelve una lista de entradas con:
      - titulo
      - link (URL original del artículo)
      - fecha (string)
      - description (snippet breve), si existe
    """
    all_entries = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            all_entries.append({
                "titulo": entry.title,
                "link": entry.link,
                "fecha": entry.get("published", entry.get("updated", "")),
                # description podría venir como entry.summary o entry.description
                "snippet": entry.get("summary", entry.get("description", ""))  
            })
    return all_entries

if __name__ == "__main__":
    noticias = fetch_all_feeds(RSS_FEEDS)
    print(f"🔍 Encontré {len(noticias)} entradas en total.\n")
    for i, n in enumerate(noticias, start=1):
        print(f"{i}. {n['titulo']}")
        print(f"   URL: {n['link']}")
        print(f"   Fecha: {n['fecha']}")
        print(f"   Snippet: {n['snippet'][:80]}...\n")
    # Guardamos mínima info en JSON para el próximo paso
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    with open("links.json", "w", encoding="utf-8") as f:
        json.dump(noticias, f, indent=2, ensure_ascii=False)
    print(f"✅ Guardé todo en links.json")