<h1># Twitter Scraping (with Snscrape) & Sentiment Analysis</h1>
by: Samuel González

<h2><b>Introducción</b></h2>
</br>
Twitter es la red social de microblogging más importante en Colombia. A pezar de la reciente entrada de <b>Mastodon</b>, Twitter se mantiene como líder con más de 2MM de tweets al día, 2 millones de usuarios activos y una cantidad de temas de los que se discute y que luego se expanden hacia otros medios, volviendose importante el análisis de estas temáticas que terminan modelando nuestras realidades. 
</br></br>

"Como es arriba, es abajo", dice una de las leyes herméticas. Como es en el microcosmos Twittero, es en el microcosmos Político. A esto me refiero cuando hablo de moldear realidades. Si analizamos lo que pasa en Twitter, podremos tener una idea de como se moverán las conversaciones en otros ambientes sociales, teniendo cuidado con los sesgos que pudieran surgir.
</br></br>

Este ha sido un proyecto personal que llevo trabajando desde hace meses, cuando comencé mi Master en Ciencia de Datos. Espero pueda servir como punto de partida para otras personas que quieran hacer Análisis de Sentimientos con Tweets. También espero obtener feedback, acerca de los posibles sesgos, mejoras de performance, entre otros.
</br>
</br>


<h2><b>Librerías utilizadas</b></h2>
</br>
Este algoritmo tiene como base las librerías de Pandas, Snscrape y PYSentimiento, entre otras que se mencionan más abajo, pero son estas las más relevantes. 
</br></br>

El scraping fue realizado con Snscrape, mediante las funciones de búsqueda en el Feed y por Usuario. La documentación de esta librería es fácilmente accesible y tiene un buen soporte y actualización.
</br></br>

La manipulación de los datos en general se realiza con Pandas, en algunos casos se usa Numpy para funciones básicas. Fue necesario un proceso amplio de limpieza y preprocesamiento, donde se usaron librerías alternas como NLTK, re, string, itertools.
</br></br>

El análisis de sentimiento se trabajó con la librería PYSentimiento, una librería muy útil para este tipo de proyectos en español. En general hay mucha documentación y librerías para SA en inglés, pero en español aún faltan más alternativas.
</br>
</br>
<code>from nltk.corpus import stopwords
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from nltk.tokenize import word_tokenize # to create word tokensimport nltk
from nltk.stem import WordNetLemmatizer # to reduce words to orginal form
from pysentimiento.preprocessing import preprocess_tweet
import seaborn as sns
import snscrape.modules.twitter as sntwitter
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import nltk 
import string
import re
from time import perf_counter</code>
</br>
</br>
