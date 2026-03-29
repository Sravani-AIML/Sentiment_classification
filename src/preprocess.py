
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

#TF-IDF(term frequency and inverse documnet frequency)
def clean_text_tfidf(text):
  """
  clean each review
  Lowercase
  remove urls
  remove html tags
  remove punctuations
  remove stopwords
  apply stemming
  """
  text = text.lower() #lowercase
  text = re.sub(r'http\S+|www\S+',"", text) #remove urls
  text = re.sub(r'<.*?>','',text) #remove html tags
  text = re.sub(r'[^A-Za-z0-9\s]','',text) #remove punctuations

  # stemming + stopwords removal
  tokens = [
      stemmer.stem(token)
      for token in word_tokenize(text)
      if token not in stop_words
  ]
  
  return " ".join(tokens)

#Embedding with no stemming and stopword removal
def clean_text_embedding(text):
  text = text.lower() #lowercase
  text = re.sub(r"https\S+|www\S+","",text) #remove urls
  text = re.sub(r"<.*?>","",text) #remove html tags
  text = re.sub(r"[^A-Za-z0-9\s]","",text) #remove punctuations

  return text


def load_and_preprocess(df_path, text_column, label_column, mode = "tfidf"):
  """
  Load the Dataset and apply preprocessing
  parameters 
  - df_path : path to csv dataset file,
  - text_column : column containing string data
  - label_column : column containing labels
  - mode:
    "tfidf" -> applies stemming + stopword removal (for TF-IDF models)
    "embedding" -> keeps full preprocessed text without stemming (for Embedding models)

  returns
  - X: Preprocessed text
  - y: Encoded labels(0/1)
  """
  df = pd.read_csv(df_path)
  if mode == "tfidf":
    #Preprocessing optimized for tfidf (bag of words)
    df[text_column] = df[text_column].apply(clean_text_tfidf)
  elif mode == "embedding":
    #Preprocessing optimized for embedding 
    df[text_column] = df[text_column].apply(clean_text_embedding)
  else:
    raise ValueError("mode should be tfidf or embedding")

  #text values
  X = df[text_column]
  #Encoded labels
  
  y = LabelEncoder().fit_transform(df[label_column])
  return X, y

