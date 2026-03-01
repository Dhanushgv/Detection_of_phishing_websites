# src/lstm_model_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLSTM(nn.Module):
    """
    Character-level LSTM classifier.
    Embedding -> BiLSTM -> Attention-pooling (or max-pool) -> FC -> Sigmoid
    Returns probability in [0,1].
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.5, bidirectional: bool = True):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # simple attention: learnable vector
        self.attn_w = nn.Linear(hidden_dim * self.num_directions, 1)
        fc_in = hidden_dim * self.num_directions
        self.fc1 = nn.Linear(fc_in, 128)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs, _ = self.lstm(emb)  # outputs: (batch, seq_len, hidden_dim * num_directions)

        # attention pooling over time
        attn_scores = self.attn_w(outputs).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        weighted = (outputs * attn_weights).sum(dim=1)  # (batch, hidden_dim * num_directions)

        x = F.relu(self.fc1(weighted))
        x = self.dropout(x)
        x = torch.sigmoid(self.out(x)).squeeze(1)  # (batch,)
        return x
