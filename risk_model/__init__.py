import torch.nn as nn

class classifier(nn.Module):
  #define all the layers used in model
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
    #Constructor
    super().__init__()          
        
    #embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    #lstm layer
    self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
    #dense layer
    self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    #activation function
    self.act = nn.Sigmoid()
        
  def forward(self, text, text_lengths):
        
    #text = [batch size,sent_length]
    embedded = self.embedding(text)
    #embedded = [batch size, sent_len, emb dim]
      
    #packed sequence
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
  
    packed_output, (hidden, cell) = self.lstm(packed_embedded)
    #hidden = [batch size, num layers * num directions,hid dim]
    #cell = [batch size, num layers * num directions,hid dim]
        
    #concat the final forward and backward hidden state
    hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
    #hidden = [batch size, hid dim * num directions]
    dense_outputs=self.fc(hidden)

    #Final activation function
    outputs=self.act(dense_outputs)
        
    return outputs

"""
class Attention(nn.Module):
  def __init__(self, input_embed_dim, source_out_dim, output_embed_dim):
    super().__init__()
        
    self.input_embed_dim = input_embed_dim # decoder embedding dim
    self.source_out_dim = source_out_dim # encoder outputs length
    self.output_embed_dim = output_embed_dim # attention output dim
        
    # self.output_proj should take in the concatenation of embedded decoder input and weighted encoder outputs 
    # and output a vector with attention output dim
    self.output_proj = nn.Linear(self.input_embed_dim+self.source_out_dim, self.output_embed_dim)

  def forward(self, input, source_hids):
    output = None
    # ===============================
    # attention takes in "input" -- embedded decoder input
    # and "source_hids" -- encoder outputs
    # it first calculated the attnetion weights by softmaxing the multiplication result of input and source_hids 
    # then calculate the weighted sum of encoder outputs
    # finally concate the weighted sum and the embedded decoder input and pass through a linear function to reduce dimentionality
    # before returing the output, pass is through a tanh function
    # ===============================
    # multiplication result of input and source_hids
    attention_weights = (source_hids * input.unsqueeze(0)).sum(dim=2)
    # softmaxing result
    attention_weights = F.softmax(attention_weights, dim=0)
    # calculate the weighted sum of encoder outputs
    weighted_sum = (attention_weights.unsqueeze(2) * source_hids).sum(dim=0)
    # concate the weighted sum and the embedded decoder input
    weighted_sum = torch.cat((weighted_sum, input), dim=1)
    # pass through a linear function
    output = self.output_proj(weighted_sum)
    # pass is through a tanh function
    output = torch.tanh(output)
    return output
"""



