

from nltk.corpus import stopwords
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
from time import perf_counter




t1_start = perf_counter()
# Definir una lista con las palabras clave
keywords = ['shakira']
# Definir una variable para la ubicación
loc = '4.570868, -74.297333, 600km'
# Iterar sobre la lista de palabras clave
for keyword in keywords:
# Scrapear los datos
    df = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
        '{} since:2022-12-01 until:2023-01-10 lang:"es" geocode:"{}"'.format(keyword, loc)).get_items(), 2500))[['date',
      'id',
      'rawContent', 
      'replyCount',
      'retweetCount',
      'likeCount',
      'quoteCount',
      'user',
      'conversationId',
      'sourceLabel',
      'retweetedTweet',
      'quotedTweet',
      'inReplyToTweetId',
      'inReplyToUser',
      'mentionedUsers',
      'coordinates',
      'place',
      'viewCount']]
             # Guardar los datos en un archivo
    df.to_csv('{}.csv'.format(keyword))
            # Extraer los datos de la columna user
    user_data = df['user'].to_list()
    # Crear un nuevo dataframe desde los diccionarios de la columna user
    DF_user = pd.DataFrame.from_dict(user_data)
# Unir el nuevo dataframe al dataframe original
df_final = pd.concat([df, DF_user], axis=1)
df_final.to_csv('raw_scraping.csv', index=False)
t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)


##lista de los archivos csv
#lista_csv = ["sativa.csv", 
#             "indica.csv",
#             "marihuana_10000.csv",
#             "marihuana_10000_2.csv",
#             "marihuana_10000_3.csv",
#             "mariguana.csv",
#             "weed.csv",
#             "marihuanero.csv",
#             "marihuanera.csv",
#             "cannabis.csv",
#             "porro.csv"]

#crear un dataframe vacío
#df_final = pd.DataFrame()

#unir los dataframes de todos los archivos csv
#for archivo in lista_csv:
 #   df = pd.read_csv(archivo)
  #  df_final = pd.concat([df_final, df], axis=0)

#mostrar el contenido del dataframe final
#print(df_final)

#df_final.to_csv('final_concat_.csv')


t1_start = perf_counter()
#CLEANING
df = pd.DataFrame(pd.read_csv('raw_scraping.csv')) 
#replacing Nan "location" with "No loaction"
df["location"] = df["location"].fillna("NoLocation")
#Remove columns
#df.drop(df.columns[[2,5,7,8,18,19,20,21,22,23]], axis=1, inplace=True)
#Ajustar booleanos
df['verified'] = np.where(df['verified'] == 'True', True, False)
df["date"] = df["date"].apply(pd.to_datetime)
#reply_to_user column
# Crea la nueva columna y asigna el valor 'no_reply' a todas las filas
df['reply_to_user'] = 'no_reply'
# Usa una expresión lambda para buscar los tweets que inician con una arroba
# y asignarle el username a la columna 'reply_to_user'
df.loc[df['rawContent'].str.startswith('@'), 'reply_to_user'] = df['rawContent'].apply(lambda x: x.split()[0])
#Limpiar Location
#define a function to clean location and removie unwanted characters
emojis = list(UNICODE_EMOJI.keys())
def processing_location(location):
    location = re.sub(r'\@\w+|\#\w+|\d+', '',  location)                          # Cleaning and removing repeating characters
    location_tokens = word_tokenize(location)
    filter_words = [w for w in location_tokens if w not in emojis]
    unpunctuated_words = [w for w in filter_words if w not in string.punctuation] # Cleaning and removing punctuations
    lemmatizer = WordNetLemmatizer() 
    lemmatized_words = [lemmatizer.lemmatize(w) for w in unpunctuated_words]
    return " ".join(lemmatized_words)
# Generate a new column called 'Processed Tweets' by applying preprocessed tweets function to the 'Tweet' column.
df['cleaned_location'] = df['location'].apply(processing_location)
def standardize_location(location):
    location = location.str.replace(r'\b(Bogota\w*)\b', 'Bogota', case=False)
    location = location.str.replace(r'\b(Bogot[aáà]\s*[\/\w]*)\b', 'Bogota', case=False)
    location = location.str.replace(r'\b(Bogot[aáà]\s*[/.\w]*)\b', 'Bogota', case=False)
    location = location.str.replace(r'\b(cali\s*[\/\w]*)\b', 'Cali', case=False)
    location = location.str.replace(r'\b(medellin|medellín)\b', 'Medellin', case=False)
    location = location.str.replace(r'\b(\w+)( colombia|-colombia|. colombia)\b', r'\1', regex=True, case=False, flags=re.I)
    location = location.str.replace('colombia', 'Colombia')
    location = location.str.replace('COLOMBIA', 'Colombia')
    return location
df['cleaned_location'] = standardize_location(df['cleaned_location'])
cities = df["cleaned_location"].unique().tolist()
df.drop(df[df['cleaned_location'] == ''].index, axis=0, inplace=True)
#GRAFICA DE LOCATIONS
# Count the number of occurrences of each value in the cleaned_location column
location_counts = df['cleaned_location'].value_counts()
# Sort the location counts in descending order
sorted_location_counts = location_counts.sort_values(ascending=False)
# Select only the rows with a count greater than or equal to 1000
filtered_location_counts = sorted_location_counts.loc[sorted_location_counts >= 50]
# Create a bar chart of the filtered location counts
filtered_location_counts.plot(kind='bar')
# Show the chart
plt.show()
df.to_csv('cleaned_scraping.csv')
t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)





#ANALISIS DE USERS SEGUN FOLLOWERS Y LIMPIEZA DE USERS QUE NO APORTAN
#Filtrar los users y ver cantidad de followers
# Crea un nuevo dataframe con las columnas 'username' y 'followersCount' del dataframe 'df'
df_followers = pd.concat([df['username'], df['followersCount']], axis=1)
df['followersCount'] = df['followersCount'].astype(int)
df_followers.sort_values(by='followersCount', ascending=False, inplace=True)
df_followers = df_followers.drop_duplicates()
counts = df['reply_to_user'].str.replace('@', '').value_counts()
df_followers['replyCount'] = df_followers['username'].map(counts)
counts_2 = df[df['username'].isin(df_followers['username'])]['username'].value_counts()
df_followers['tweet_count'] = df_followers['username'].map(counts_2)
# Crea un diccionario con los usernames como claves y los followers como valores
username_followers = df_followers.set_index('username')['followersCount'].to_dict()
#FILTRAR NOTICIAS
# Filtra las filas que tienen 'mmgcco' o 'lucho10821' en la columna 'username'
filtered_df = df.loc[df['username'].isin(['Minvivienda','lanacionweb','infopresidencia','LaMega','RevistaDinero','QhuboCali','nuevodiaibague','LaRazonCo','ElNuevoSiglo','PublimetroCol','pulzo','rcnmundo','estoescambio','kienyke','RedMasNoticias','vanguardiacom','CABLENOTICIAS','larepublica_co','elpaiscali','Telemedellin','elcolombiano','VickyDavilaH','Citytv','NoticiasUno','Portafolioco','ELTIEMPO','rcnradio','RevistaSemana','NoticiasRCN', 'ElTiempo', 'elespectador', 'NoticiasCaracol', 'WRadioColombia', 'CaracolRadio', 'BluRadioCo', 'lafm'])]
# Elimina esas filas del dataframe original
df = df.drop(filtered_df.index)
# Genera la nube de palabras
wordcloud = WordCloud(width=800, height=400, max_words=100)
wordcloud.generate_from_frequencies(username_followers)
# Muestra el gráfico
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Usuarios con mayor cantidad de followers")
plt.show()






#defining a function to extract words with hashtags
def Hashtag(rawContent):
    rawContent = rawContent.lower()  #converts tweet to lower case
    rawContent = re.findall(r'\#\w+',rawContent)  
    return " ".join(rawContent)
#defining the column
df['hashtags'] = df['rawContent'].apply(Hashtag)
#converting the hashtag column to a list in other to get the value counts
hashtag_list = df["hashtags"].to_list()
new_hashtags = []
for hash in hashtag_list:
    new_hash = hash.split()
    for new in new_hash:
        new_hashtags.append(new)
        new_hashtags
from collections import Counter
counts = Counter(new_hashtags)
hashtag_df = pd.DataFrame.from_dict(counts, orient="index").reset_index()
hashtag_df.columns = ["hashtag", "counts"]
hashtag_df.sort_values("counts", ascending=False, inplace=True)
hashtag_df.head()
#saving hashtag count to csv file for visualization
hashtag_df.to_csv("hashtags.csv")
#WORDCLOUD HASHTAGS
# Crea una lista de palabras a partir de los hashtags en tu conjunto de datos
hashtag_words = [hashtag for hashtag in hashtag_df['hashtag']]
# Crea una cadena a partir de la lista de hashtags
hashtags_string = ' '.join(hashtag_words)
# Genera el wordcloud a partir de la cadena de hashtags
wordcloud = WordCloud(width=800, height=800, min_font_size=15).generate(hashtags_string)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 



t1_start = perf_counter()
#PREPROCESSING
#defining processed content by removing words that do not contribute to any analysis.
stop_words = list(stopwords.words('spanish'))
user_stop_words = ["emoji","url","usuario","cara","algún","alguna","algunas","alguno","algunos","ambos","ampleamos","ante","antes","aquel","aquellas","aquellos","aqui","arriba","atras","bajo","bastante","bien","cada","cierta","ciertas","cierto","ciertos","como","con","conseguimos","conseguir","consigo","consigue","consiguen","consigues","cual","cuando","dentro","desde","donde","dos","el","ellas","ellos","empleais","emplean","emplear","empleas","empleo","en","encima","entonces","entre","era","eramos","eran","eras","eres","es","esta","estaba","estado","estais","estamos","estan","estoy","fin","fue","fueron","fui","fuimos","gueno","ha","hace","haceis","hacemos","hacen","hacer","haces","hago","incluso","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","la","largo","las","lo","los","mientras","mio","modo","muchos","muy","nos","nosotros","otro","para","pero","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","por","por qué","porque","primero","puede","pueden","puedo","quien","sabe","sabeis","sabemos","saben","saber","sabes","ser","si","siendo","sin","sobre","sois","solamente","solo","somos","soy","su","sus","también","teneis","tenemos","tener","tengo","tiempo","tiene","tienen","todo","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","tuyo","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","va","vais","valor","vamos","van","vaya","verdad","verdadera","verdadero","vosotras","vosotros","voy","yo","él","ésta","éstas","éste","éstos","última","últimas","último","últimos","a","añadió","aún","actualmente","adelante","además","afirmó","agregó","ahí","ahora","al","algo","alrededor","anterior","apenas","aproximadamente","aquí","así","aseguró","aunque","ayer","buen","buena","buenas","bueno","buenos","cómo","casi","cerca","cinco","comentó","conocer","consideró","considera","contra","cosas","creo","cuales","cualquier","cuanto","cuatro","cuenta","da","dado","dan","dar","de","debe","deben","debido","decir","dejó","del","demás","después","dice","dicen","dicho","dieron","diferente","diferentes","dijeron","dijo","dio","durante","e","ejemplo","ella","ello","embargo","encuentra","esa","esas","ese","eso","esos","está","están","estaban","estar","estará","estas","este","esto","estos","estuvo","ex","existe","existen","explicó","expresó","fuera","gran","grandes","había","habían","haber","habrá","hacerlo","hacia","haciendo","han","hasta","hay","haya","he","hecho","hemos","hicieron","hizo","hoy","hubo","igual","indicó","informó","junto","lado","le","les","llegó","lleva","llevar","luego","lugar","más","manera","manifestó","me","mediante","mencionó","mi","misma","mismas","mismo","mismos","momento","mucha","muchas","mucho","nada","nadie","ni","ningún","ninguna","ningunas","ninguno","ningunos","no","nosotras","nuestra","nuestras","nuestro","nuestros","nueva","nuevas","nuevo","nuevos","nunca","o","ocho","otra","otras","otros","parece","parte","partir","pasada","pasado","pesar","poca","pocas","poco","pocos","podrá","podrán","podría","podrían","poner","posible","próximo","próximos","primer","primera","primeros","principalmente","propia","propias","propio","propios","pudo","pueda","pues","qué","que","quedó","queremos","quién","quienes","quiere","realizó","realizado","realizar","respecto","sí","sólo","se","señaló","sea","sean","según","segunda","segundo","seis","será","serán","sería","sido","siempre","sigue","siguiente","sino","sola","solas","solos","son","tal","tampoco","tan","tanto","tenía","tendrá","tendrán","tenga","tenido","tercera","toda","todas","todavía","todos","total","trata","través","tres","tuvo","usted","varias","varios","veces","ver","vez","y","ya"]
#Palabras personalizadas
words = ['shakira']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + user_stop_words + alphabets + words
emojis = list(UNICODE_EMOJI.keys())
nltk.download('punkt')
def processing_tweets(rawContent):
    rawContent = preprocess_tweet(rawContent)  # Process tweet with pysentimiento's preprocess_tweet function
    rawContent = rawContent.lower()
    rawContent = re.sub(r'\@\w+|\#\w+|\d+|\.\.\.|\'|\"|¿', '', rawContent)                      # Cleaning and removing repeating characters
    tweet_tokens = word_tokenize(rawContent)  
    filter_words = [w for w in tweet_tokens if w not in stop_words]
    filter_words = [w for w in filter_words if w not in emojis]
    unpunctuated_words = [w for w in filter_words if w not in string.punctuation] # Cleaning and removing punctuations
    lemmatizer = WordNetLemmatizer() 
    lemmatized_words = [lemmatizer.lemmatize(w) for w in unpunctuated_words]
    return " ".join(lemmatized_words)
# create a new column called 'cleaned tweets' by applying processing tweets function to the tweet column.
df['cleaned_tweets'] = df['rawContent'].apply(processing_tweets)
df.head()
t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)




#GRÁFICA WORDCLOUD
stopwords = set(STOPWORDS) 
words = ''
for cleaned_tweets in df.cleaned_tweets:
    tokens = str(cleaned_tweets).split()
    tokens = [i.lower() for i in tokens]
    
    words += ' '.join(tokens) + ' '
sns.set_style('dark')
colormap = sns.color_palette("Reds", 8)   
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 15).generate(words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 






#Grafico de tweets por minuto. Freq: D, W, M, Q, H, #min, S
tweet_time_df = df.groupby(pd.Grouper(key='date', freq='30min', convention='start')).size()
tweet_time_df.plot(figsize=(18,6))
plt.ylabel("Per Day Tweet Count")
plt.title("Tweets (Day)")
plt.grid(True)

#save the final dataframe to csv 
df.to_csv("final_scraping.csv", index=False)




#seleccionar solamente un período, en caso que hayan tweets muy viejos que no aporten al análisis, o se quiera centrar en un período específico
#df_1 = df.iloc[:637,:]
#df_2 = df.iloc[638:,:]

#última limpieza para mejorar el sentiment
#Eliminar todas las filas que tengn solo 1 o 2 palabras
df = df.dropna(subset=['cleaned_tweets'])
df = df[df['cleaned_tweets'].str.split().apply(len) > 2]
#Eliminar todo lo que sea reply
df = df.loc[df['reply_to_user'] == 'no_reply']


#SENTIMENT ANALYSIS
t1_start = perf_counter()
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")
pysentimiento_scraping = analyzer.predict(df['cleaned_tweets'])
df_sentiment_raw = pd.DataFrame(pysentimiento_scraping)
df_sentiment_raw.to_csv("pysentimiento_raw.csv", index=False)


#Limpiar el Sentiment
df = pd.DataFrame(pd.read_csv('pysentimiento_raw.csv'))
#Quitar texto AnalyzerOutput
df = df.replace('AnalyzerOutput', '', regex=True)
#Quitar parentesis
df = df.apply(lambda x: x.str.replace('(','').str.replace(')',''))
#Cambiar nombre de la columna 0 para facilitar llamarla
df.rename(columns = {'0':'raw'}, inplace = True)
#Desglozar la columna en dos
df[['output', 'probas']] = df['raw'].str.split(',', expand=True, n=1)
#Eliminar la primer columna
df.drop('raw', axis=1, inplace=True)
#eliminar output= de la segunda columna
df = df.replace('output=', '', regex=True)
#eliminar probas=
df = df.replace('probas=', '', regex=True)
#eliminar llaves
df = df.apply(lambda x: x.str.replace('{','').str.replace('}',''))
#desglosar la columna probas
df[['prob_1', 'prob_2', 'prob_3']] = np.array(df['probas'].str.split(',').tolist())
#drop probas
df.drop('probas', axis=1, inplace=True)
df.to_csv("pysentimiento_cleaned.csv", index=False)
df = df.astype('string')
df.groupby('output').size().plot.pie()
plt.title("Sentiment Analysis")
plt.grid(True)



#Promedio de las probas, pero antes debo quitarle el texto a todas y convertirlo a float
#df.loc[:, ['NEG', 'NEU', 'POS']].mean().plot.pie()




#Necesitamos ahora unir el cleaned_scraping con este sentiment cleaned
df = pd.DataFrame(pd.read_csv('final_scraping.csv'))
df1 = pd.DataFrame(pd.read_csv('pysentimiento_cleaned.csv'))
df_concat = pd.concat([df, df1], axis=1)
df.to_csv("pysentimiento_cleaned_concat.csv", index=False)
#cambiar de lugar la columna cleaned tweets que quedó muy lejos
col_cleanedtweets = df_concat.columns[15]
# Crear una lista con las columnas en el orden deseado
columns_reordered = [col for col in df_concat.columns if col != col_cleanedtweets]
columns_reordered.insert(2, col_cleanedtweets)
# Reordenar el DataFrame utilizando la lista de columnas
df_concat = df_concat.reindex(columns=columns_reordered)
#Filtrar solo los tweets positivos
DF_pos = pd.DataFrame(df_concat[df_concat['output'] == 'POS'])
#Filtrar solo los tweets negativos
DF_neg = pd.DataFrame(df_concat[df_concat['output'] == 'NEG'])
#Filtrar solo los tweets neutros
DF_neu = pd.DataFrame(df_concat[df_concat['output'] == 'NEU'])
#Limpiar las columnas de PROBAS
df_concat['prob_1'] = df_concat['prob_1'].astype(str)
df_concat['label_1'] = df_concat['prob_1'].apply(lambda x: x.split(':')[0])
df_concat['probability_1'] = df_concat['prob_1'].apply(lambda x: float(x.replace('NEU:','').replace('NEG:','').replace('POS:','')))
df_concat['prob_2'] = df_concat['prob_2'].astype(str)
df_concat['label_2'] = df_concat['prob_2'].apply(lambda x: x.split(':')[0])
df_concat['probability_2'] = df_concat['prob_2'].apply(lambda x: float(x.replace('NEU:','').replace('NEG:','').replace('POS:','')))
df_concat['prob_3'] = df_concat['prob_3'].astype(str)
df_concat['label_3'] = df_concat['prob_3'].apply(lambda x: x.split(':')[0])
df_concat['probability_3'] = df_concat['prob_3'].apply(lambda x: float(x.replace('NEU:','').replace('NEG:','').replace('POS:','')))
df_concat = df_concat.drop(labels=['prob_1', 'prob_2', 'prob_3'], axis=1)
# Group the data by label and calculate the mean probability for each column
probs = df_concat.groupby('label_1')['probability_1', 'probability_2', 'probability_3'].mean()
# Create a bar plot
probs.plot(kind='bar')
# Add a title and show the plot
plt.title('Average Probabilities by Label')
plt.show()
t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)







#SENTIMENT ANALYSIS NEU
t1_start = perf_counter()
analyzer = create_analyzer(task="sentiment", lang="es")
pysentimiento_scraping_neu = analyzer.predict(DF_neu['rawContent'])
df_sentiment_raw_neu = pd.DataFrame(pysentimiento_scraping_neu)

df_sentiment_raw_neu.to_csv("df_sentiment_raw_neu.csv", index=False)

#Limpiar el Sentiment

df_sentiment_neu = pd.DataFrame(pd.read_csv('df_sentiment_raw_neu.csv'))
#Quitar texto AnalyzerOutput
df_sentiment_neu = df_sentiment_neu.replace('AnalyzerOutput', '', regex=True)
#Quitar parentesis
df_sentiment_neu = df_sentiment_neu.apply(lambda x: x.str.replace('(','').str.replace(')',''))
#Cambiar nombre de la columna 0 para facilitar llamarla
df_sentiment_neu.rename(columns = {'0':'raw'}, inplace = True)
#Desglozar la columna en dos
df_sentiment_neu[['output', 'probas']] = df_sentiment_neu['raw'].str.split(',', expand=True, n=1)
#Eliminar la primer columna
df_sentiment_neu.drop('raw', axis=1, inplace=True)
#eliminar output= de la segunda columna
df_sentiment_neu = df_sentiment_neu.replace('output=', '', regex=True)
#eliminar probas=
df_sentiment_neu = df_sentiment_neu.replace('probas=', '', regex=True)
#eliminar llaves
df_sentiment_neu = df_sentiment_neu.apply(lambda x: x.str.replace('{','').str.replace('}',''))
#desglosar la columna probas
df_sentiment_neu[['prob_1', 'prob_2', 'prob_3']] = np.array(df_sentiment_neu['probas'].str.split(',').tolist())
#drop probas
df_sentiment_neu.drop('probas', axis=1, inplace=True)
df_sentiment_neu.to_csv("pysentimiento_cleaned.csv", index=False)

df_sentiment_neu = df_sentiment_neu.astype('string')
df_sentiment_neu.groupby('output').size().plot.pie()
plt.title("Sentiment Analysis")
plt.grid(True)

#Promedio de las probas, pero antes debo quitarle el texto a todas y convertirlo a float
#df.loc[:, ['NEG', 'NEU', 'POS']].mean().plot.pie()


#Necesitamos ahora unir el cleaned_scraping con este sentiment cleaned
df = pd.DataFrame(pd.read_csv('final_scraping.csv'))
df1 = pd.DataFrame(pd.read_csv('pysentimiento_cleaned.csv'))
df_concat = pd.concat([df, df1], axis=1)

df.to_csv("pysentimiento_cleaned_concat.csv", index=False)

#cambiar de lugar la columna cleaned tweets que quedó muy lejos
col_cleanedtweets = df_concat.columns[15]
# Crear una lista con las columnas en el orden deseado
columns_reordered = [col for col in df_concat.columns if col != col_cleanedtweets]
columns_reordered.insert(2, col_cleanedtweets)
# Reordenar el DataFrame utilizando la lista de columnas
df_concat = df_concat.reindex(columns=columns_reordered)

#Filtrar solo los tweets positivos
DF_pos_2 = pd.DataFrame(df_concat[df_concat['output'] == 'POS'])
#Filtrar solo los tweets negativos
DF_neg_2 = pd.DataFrame(df_concat[df_concat['output'] == 'NEG'])
#Filtrar solo los tweets neutros
DF_neu_2 = pd.DataFrame(df_concat[df_concat['output'] == 'NEU'])

#Limpiar las columnas de PROBAS
df_concat['prob_1'] = df_concat['prob_1'].astype(str)
df_concat['label_1'] = df_concat['prob_1'].apply(lambda x: x.split(':')[0])
df_concat['probability_1'] = df_concat['prob_1'].apply(lambda x: float(x.replace('NEU:','').replace('NEG:','').replace('POS:','')))
df_concat['prob_2'] = df_concat['prob_2'].astype(str)
df_concat['label_2'] = df_concat['prob_2'].apply(lambda x: x.split(':')[0])
df_concat['probability_2'] = df_concat['prob_2'].apply(lambda x: float(x.replace('NEU:','').replace('NEG:','').replace('POS:','')))
df_concat['prob_3'] = df_concat['prob_3'].astype(str)
df_concat['label_3'] = df_concat['prob_3'].apply(lambda x: x.split(':')[0])
df_concat['probability_3'] = df_concat['prob_3'].apply(lambda x: float(x.replace('NEU:','').replace('NEG:','').replace('POS:','')))
df_concat = df_concat.drop(labels=['prob_1', 'prob_2', 'prob_3'], axis=1)

# Group the data by label and calculate the mean probability for each column
probs = df_concat.groupby('label_1')['probability_1', 'probability_2', 'probability_3'].mean()
# Create a bar plot
probs.plot(kind='bar')
# Add a title and show the plot
plt.title('Average Probabilities by Label')
plt.show()

#Aquí me están quedando unos Nan

t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)


#Unir los DFs por PROB
df_neu = pd.concat([DF_neu, DF_neu_2], axis=0)
df_pos = pd.concat([DF_pos, DF_pos_2], axis=0)
df_neg = pd.concat([DF_neg, DF_neg_2], axis=0)











