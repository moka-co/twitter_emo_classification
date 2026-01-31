import torch
import torch.nn as nn
import torch.nn.functional as F

class EmoBaseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, glove_weights, distributions=None):
        super(EmoBaseline, self).__init__()

        self.distributions = distributions # Prior initialization

        # 1. Embedding Layer
        self.embedding = nn.Embedding.from_pretrained(
            glove_weights,
            freeze=False
        )

        # 2. MLP Layers
        # Since we are averaging embeddings, the input to the first linear layer
        # is just embedding_dim (not hidden_dim * 2)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        # text shape: [batch_size, seq_len]

        # 1. Convert tokens to embeddings
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]

        # 2. SIMPLE AVERAGING (The "Baseline" part)
        # We average across the sequence length (dim=1)
        # pooled shape: [batch_size, embedding_dim]
        pooled = torch.mean(embedded, dim=1)

        # 3. Pass through MLP
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    # Apply Xavier initialization to linear layer
    # and prior initialization to last layer
    def init_weights(self):
        # Calculate prior biases from distributions
        prior_biases = None
        if self.distributions is not None:
            freqs = torch.tensor(self.distributions, dtype=torch.float)
            # Add epsilon to prevent log(0)
            prior_biases = torch.log(freqs + 1e-9)

        for name, param in self.named_parameters():
            # Skip embedding layer
            if 'embedding' in name:
                continue

            # Xavier for all weights
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)

            # Bias Initialization
            elif 'bias' in name:
                # Apply prior to the final layer (fc2)
                if name == 'fc2.bias' and prior_biases is not None:
                    param.data.copy_(prior_biases)
                else:
                    param.data.fill_(0)