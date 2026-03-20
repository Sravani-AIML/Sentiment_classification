
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

stop_words = stopwords.words('english')
stemmer = PorterStemmer()


def clean_text(text):
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

def load_and_preprocess(df_path, text_column, label_column):
  #load the data
  df = pd.read_csv(df_path)

  #clean text
  df[text_column] = df[text_column].apply(clean_text)

  #text values
  X = df[text_column]
  #Encoded labels
  
  y = LabelEncoder().fit_transform(df[label_column])
  return X, y

