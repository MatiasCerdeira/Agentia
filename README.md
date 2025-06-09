# Agentia

La idea de este primer repo es hacer una prueba rapida de la arquitectura RAG como solucion del problema del Clipping Diario.

Despues de descargar el repo hay que correr el siguiente codigo para instalar las dependencias:

```python
python3 -m venv venv
source venv/bin/activate  # en macOS/Linux
.\venv\Scripts\activate   # en Windows

pip install -r requirements.txt
```

Lo de Windows no estoy tan seguro igual, habria que chequear.

Para ejecutar el programa hay que hacer un python main.py
Es importante que antes de hacer eso hayas activado el venv.

## Que hace el programa

### Fetch_links

- Busca los articulos del dia en algunos medios (se puede ver la lista en main.py->RSS_FEEDS)
- Utiliza RSS para obteners los links de los articulos.

### Fetch_full_articles

- Utiliza la libreria newspaper3k para scrapear y parsear los articulos.
- Guarda la lista de articulos completos en 2 lugares:
  - articulos_completos.json
  - la carpeta articulos_txt --> aca cada articulo es su propio txt

### Build_faiss

- Rompe los articulos en chunks.
- Vectoriza cada chunk.
- Utiliza una Base de Datos Vectorial Local.
- Genera dos archivos: mapping_id2chunk.pkl y noticias_politica.index

### Query_rag

- Busca los chunks que mas se acercan al query dado

## Problemas

- Por ahora el resultado es malisimo. Pareciera traer los chunks que mas dicen "politica", en vez de los mas relevantes.

## To Do

- Hay que ver de hacer clustering o una regresion logistica o algo similar.
- Una vez que tenemos los clusters podemos pasar como un resumen de cada cluster a GPT y dejar que este tome la decision de que es relevante.
- Cuando sabemos que es relevante, le vamos pasando toda la info de cada cluster a GPT (en distintos chats) y hacemos que escriba el resumen y el analisis (y lo retorne como JSON usando Structured Outputs)

## Tener en cuenta

- Probablemente haya que modificar la forma en la que estamos creando los chunks porque no parece optima.
- Hay que ver sobre que se hace el clustering. Hacerlo sobre los chunks no parece tener sentido. Pero no usar chunks empeora el resultado porque pierde mucha granuralidad y detalle.
