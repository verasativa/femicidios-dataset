import numpy as np
import pandas as pd
from scipy import stats
import nltk, re, random

def bi_modal(mode_a=.2, mode_b=.8, sd=.19, size=100):
    data = []
    for i in range(size):
        modal = random.randint(0,1)
        if modal == 0:
            data.append(random.normalvariate(.8, sd))
        if modal == 1:
            data.append(random.normalvariate(.2, sd))
    return data

def display_locations(array):
    estimations = {}
    estimations['mean'] = np.mean(array)
    estimations['trim_mean'] = stats.trim_mean(array, 0.1)
    estimations['median'] = np.median(array)
    
    for name, value in estimations.items():
        display('{}: {}'.format(name, value))        

# ¿Curiosa con la siguiente función?
# Intenta algo así:
# words = func.text_analisys(femicidios.Hecho)
# top_words = words.sort_values(by='usos', ascending=False)[:20].set_index('palabra')
# top_words.plot.barh()

def text_analisys(series):
    """Algo de procesado de lenguaje natural.
    Intenta algo así:
    words = func.text_analisys(femicidios.Hecho)
    top_words = words.sort_values(by='usos', ascending=False)[:20].set_index('palabra')
    top_words.plot.barh()
    """
        
    # Solo palabras
    series = series.dropna()
    corpus = ' '.join(series)
    symbols = [';', ',', '.', ':', '(', ')', "''", '&']
    for symbol in symbols:
        corpus = corpus.replace(symbol, '')
    # Lista de palabras
    words = nltk.word_tokenize(corpus.lower())
    
    # Preparamos herramientas y diccionario de resultados para el loop
    stemmer = nltk.stem.snowball.SnowballStemmer('spanish')
    stopwords = nltk.corpus.stopwords.words('spanish')  
    word_frequencies = {}
    for word in words:
        word = stemmer.stem(word)
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    return pd.DataFrame(word_frequencies.items(), columns=['palabra', 'usos'])