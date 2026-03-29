  
  
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size= 128, num_layers = 1):
    super().__init__()

    #Embedding layer
    self.embed = nn.Embedding(vocab_size, embedding_size)

    #Gated recurrent layer
    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,batch_first = True)

    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    #pass input to embedding layer
    x = self.embed(x)

    #pass embedded input to lstm 
    outputs , (h_n, C_n) = self.lstm(x)

    #predict the probabilties from fully connected layer
    out = self.fc(outputs[:,-1,:])

    return out

class GRUModel(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size= 128, num_layers = 1):
    super().__init__()

    #Embedding layer
    self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)

    #Gated recurrent layer
    self.gru = nn.GRU(embedding_size, hidden_size, num_layers,batch_first = True)

    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    #pass input to embedding layer
    x = self.embed(x)

    #pass embedded input to lstm 
    outputs , h_n = self.gru(x)

    #predict the probabilties from fully connected layer
    out = self.fc(outputs[:,-1,:])

    return out

class BIGRUModel(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size= 128, num_layers = 1):
    super().__init__()
    self.hidden_size = hidden_size
    #Embedding layer
    self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)

    #Gated recurrent layer
    self.gru = nn.GRU(embedding_size, hidden_size, num_layers,batch_first = True, bidirectional = True)

    #hidden_size *2 because bidirectional
    self.fc = nn.Linear(hidden_size*2, 1)

  def forward(self, x):
    #pass input to embedding layer
    x = self.embed(x)

    #pass embedded input to lstm 
    outputs , h_n = self.gru(x)

    #forward hidden state at last sequence
    #hidden_features = [forward_fetures | backward_features]
    #:hidden_size - forward features, hidden_size: backward features
    forward = outputs[:,-1,:self.hidden_size]
    #backward hidden state at last sequence 0
    backward = outputs[:,0,self.hidden_size:]

    combined_hidden = torch.cat((forward, backward), dim = 1)

    #predict the probabilties from fully connected layer
    out = self.fc(combined_hidden)

    return out


class CNNModel(nn.Module):
  def __init__(self,vocab_size, embedding_size, num_filters = 100):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)

    #Convolutional Layers
    self.conv1 = nn.Conv1d(embedding_size, num_filters, kernel_size = 3)
    self.conv2 = nn.Conv1d(embedding_size, num_filters, kernel_size = 4)
    self.conv3 = nn.Conv1d(embedding_size, num_filters, kernel_size = 5)

    #Activation
    self.relu = nn.ReLU()

    #Global pooling 
    self.pool = nn.AdaptiveMaxPool1d(1)

    self.fc = nn.Linear(num_filters *3, 1)
  
  def forward(self, x):
    #batch size, sequence length, input size
    x = self.embedding(x)

    #batch size, input size, sequence length
    x = x.permute(0,2,1)

    #conv + ReLU
    x1 = self.relu(self.conv1(x))
    x2 = self.relu(self.conv2(x))
    x3 = self.relu(self.conv3(x))

    #Global max pooling
    x1 = self.pool(x1).squeeze(2)
    x2 = self.pool(x2).squeeze(2)
    x3 = self.pool(x3).squeeze(2)

    #Concatenate
    x = torch.cat((x1, x2, x3), dim = 1)

    out = self.fc(x)

    return out


class CNN_GRU_Model(nn.Module):
  def __init__(self, vocab_size, embedding_size, num_filters = 100, hidden_size = 128):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

    #convolutional layer
    self.conv1 = nn.Conv1d(embedding_size, num_filters, kernel_size=3)
  
    # GRU input size = combined filters
    self.gru = nn.GRU(
      input_size=num_filters,
      hidden_size=hidden_size,
      batch_first=True
    )
    self.relu = nn.ReLU()

    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    x = self.embedding(x)              # (batch, seq_len, embed_dim)
    x = x.permute(0, 2, 1)             # (batch, embed_dim, seq_len)

    # CNN
    x = self.relu(self.conv1(x))
   

    # (batch, num_filters*3, seq_len)

    # Prepare for GRU
    x = x.permute(0, 2, 1)
    # (batch, seq_len, num_filters*3)

    _, hidden = self.gru(x)

    out = self.fc(hidden[-1])

    return out

    
    
