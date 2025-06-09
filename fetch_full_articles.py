import json
import os
import time
from newspaper import Article

# 1. Nombre del JSON que generaste antes con todos los links
#    Cambiá el nombre si tu archivo se llama distinto, ej: "rss_politica_todos_ar_2025-06-04.json"
INPUT_JSON = "links.json"

# 2. Carpeta donde vas a guardar cada artículo completo como .txt (para tenerlos separados)
OUTPUT_DIR = "articulos_txt"

def ensure_output_dir(dir_path):
    """Crea la carpeta OUTPUT_DIR si no existe."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def load_links(input_file):
    """Carga la lista de entradas desde el JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        entradas = json.load(f)
    return entradas

def download_full_articles(entradas, output_dir):
    """
    Recorre cada entrada (que tiene 'link', 'titulo', 'fecha') y
    con newspaper3k baja el texto completo, guardándolo en un .txt
    cuyo nombre es 'articulo_XXX.txt' (para no pisar nombres raros).
    """
    total = len(entradas)
    ensure_output_dir(output_dir)
    resultados = []  # Para guardar metadata + texto si querés JSON final
    
    for idx, ent in enumerate(entradas, start=1):
        url = ent.get("link")
        titulo = ent.get("titulo", "")
        fecha = ent.get("fecha", "")
        
        print(f"🔄 [{idx}/{total}] Bajando nota: {titulo[:50]}...")
        
        try:
            art = Article(url, language="es")
            art.download()
            art.parse()
            texto = art.text  # Contenido completo del artículo
            
            # 3. Guardar en archivo .txt
            filename_txt = f"articulo_{idx:03d}.txt"
            ruta_txt = os.path.join(output_dir, filename_txt)
            with open(ruta_txt, "w", encoding="utf-8") as f_txt:
                f_txt.write(texto)
            
            # 4. (Opcional) Guardar metadata y texto en lista para un JSON final
            resultados.append({
                "id": idx,
                "titulo": titulo,
                "link": url,
                "fecha": fecha,
                "archivo_txt": ruta_txt,
                "texto": texto
            })
            
            # Mensaje de éxito
            print(f"   ✔️ Guardado en: {ruta_txt}")
        
        except Exception as e:
            print(f"   ❌ ERROR bajando {url}: {e}")
        
        # Pequeña pausa para no reventar el servidor de cada medio
        time.sleep(1)
    
    # 5. (Opcional) Guardar todos los textos y metadata en un solo JSON
    output_json = f"articulos_completos.json"
    with open(output_json, "w", encoding="utf-8") as f_json:
        json.dump(resultados, f_json, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Listo: bajé {len(resultados)} artículos completos.")
    print(f"   • Archivos .txt en carpeta: {output_dir}")
    print(f"   • JSON con todo en: {output_json}")

if __name__ == "__main__":
    # Cargamos las entradas (título, link, fecha, maybe snippet)
    entradas = load_links(INPUT_JSON)
    # Descargamos cada artículo completo
    download_full_articles(entradas, OUTPUT_DIR)