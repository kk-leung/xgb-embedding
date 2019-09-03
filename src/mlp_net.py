import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, args, num_features):
        super(MLP, self).__init__()
        # Encoder outputs = #[bs, seq_len, n_latent] #TODO: check if True
        self.args = args
        self.fnn1 = nn.Linear(num_features, args.n_latent)
        self.drop1 = nn.Dropout(p=args.dropout)
        self.fnn2 = nn.Linear(args.n_latent, args.n_latent)
        self.drop2 = nn.Dropout(p=args.dropout)
        self.fnn3 = nn.Linear(args.n_latent, 1)

    def forward(self, h):
        h = self.drop1(F.relu(self.fnn1(h)))
        h = self.drop2(F.relu(self.fnn2(h)))
        last_layer = self.fnn3(h)
        return last_layer
