from collections import Counter
from nltk.tokenize import word_tokenize
def build_vocab(texts, max_vocab_size = 20000):

  #tokenized text
  X_tokens = [word_tokenize(review) for review in texts]

  #Count the frequencies
  counter = Counter(word for tokens in X_tokens for word in tokens)

  #Select top max_vocab_size words
  X_top = counter.most_common(max_vocab_size)

  vocab = {'<PAD>': 0, '<UNK>':1}
  vocab_words = [word for word,_ in X_top ]
  vocab.update({word: idx +2 for idx, word in enumerate(vocab_words)})


  return vocab

def text_to_sequence(texts, vocab):
  
  encoded_sequences = []
  
  X_tokens = [word_tokenize(review) for review in texts]
  for tokens in X_tokens:
    sequence = []
    for word in tokens:
      if word in vocab:
        sequence.append(vocab[word])
      else:
        sequence.append(vocab['<UNK>'])
    encoded_sequences.append(sequence)
  return encoded_sequences,X_tokens

def pad_sequence(encoded_seq,seq_len = 50):
  pad_sequences =[]
  for seq in encoded_seq:
    if len(seq) < seq_len:
      seq = seq + [0]*(seq_len - len(seq))
    else:
      seq = seq[:seq_len]
    pad_sequences.append(seq)
  return pad_sequences
  
