
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features = 5000, ngram_range = (1,1)):

  """
  returns a configured tfidf vectorizer
  """
  return TfidfVectorizer(
      max_features = max_features, 
      ngram_range = ngram_range)
  

