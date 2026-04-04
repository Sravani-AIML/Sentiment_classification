import numpy as np
from gensim.models import Word2Vec

def load_w2v(tokenized_sentences, vocab, embedding_size):
  vocab_size = len(vocab)

  w2v_model = Word2Vec(
      sentences = tokenized_sentences
  )
  embedding_matrix = np.zeros((vocab_size, embedding_size))

  for word, idx in vocab.items():
    if word == "<PAD>":
      embedding_matrix[idx] = np.zeros(embedding_size)
    elif word == "<UNK>":
      embedding_matrix[idx] = np.random.normal(size = (embedding_size,))
    elif word in w2v_model.wv:
      embedding_matrix[idx] = w2v_model.wv[word]
  return embedding_matrix


def load_glove(vocab,embedding_size ):
  glove_path = "/content/drive/MyDrive/Sentiment_classification/data/glove.6B.100d.txt"
  glove_dict = {}

  with open(glove_path, encoding = "utf-8") as f:
    for line in f:
      values = line.split()
      word = values[0]
      vector = np.asarray(values[1:], dtype = "float32")
      glove_dict[word] = vector
  
  #Build embedding_matrix
  embedding_matrix = np.zeros((len(vocab), embedding_size))

  for word, idx in vocab.items():
    if word == "<PAD>":
      embedding_matrix[idx] = np.zeros(embedding_size)
    elif word == "<UNK>":
      embedding_matrix[idx] = np.random.normal(size = (embedding_size,))
    elif word in glove_dict:
      embedding_matrix[idx] = glove_dict[word]
  return embedding_matrix


