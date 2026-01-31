import torch
import torch.nn as nn
import torch.nn.functional as F

class EmoLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, glove_weights, distributions=None):
        super(EmoLSTM, self).__init__()

        self.distributions = distributions if distributions is not None else torch.ones(6)

        # Define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            glove_weights,
            freeze=False  # Set to True if you don't want to fine-tune the embeddings
        )

        # Example Bi-LSTM setup following the embedding
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.attention =  nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.init_weights()

    def forward(self, text):
        # text shape: [batch_size, seq_len]

        # 1. Convert tokens to embeddings
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]

        # 2. LSTM pass
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # 3. Compute attention
        # pass every hidden state through the attention layer to get a score
        
        energy = self.attention(lstm_output) # shape [batch_size, seq_len, 1]

        # Convert scores to weights
        weights = F.softmax(energy, dim=1) # shape [batch_size, seq_len, 1]

        # compute the context vector by linear combination of weights and lstm output
        context_vector = torch.sum(weights * lstm_output, dim=1) # shape [batch_size, hidden_dim * 2]
       
        # 4. Final Classification
        return self.fc(context_vector), weights

    # Apply Xavier initialization to LSTM and linear layer
    # and prior initialization to last layer
    def init_weights(self):

        # Calculate biases if distributions are provided
        prior_biases = None
        if isinstance(self.distributions, torch.Tensor):
            freqs = self.distributions.detach().clone()
        elif self.distributions is not None:
            freqs = torch.tensor(self.distributions, dtype=torch.float)
        else:
            freqs = None

        if freqs is not None:
            # Add epsilon to prevent log(0)
            prior_biases = torch.log(freqs + 1e-9)

        for name, param in self.named_parameters():
            # Skip the embedding layer (already loaded from GloVe)
            if 'embedding' in name:
                continue

            # Initialize LSTM weights
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                if name== 'fc.bias' and prior_biases is not None: # Prior Initialization
                  param.data.copy_(prior_biases)
                else:
                  param.data.fill_(0) # Initialize LSTM biases to zero

            # Initialize Linear layers (Attention and FC)
            if 'fc.weight' in name or 'attention.weight' in name:
                nn.init.xavier_uniform_(param.data)