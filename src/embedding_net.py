import torch
import torch.nn.functional as F


class Predictor(torch.nn.Module):
    def __init__(self, num_trees, embedding_size, hidden_size=2):
        super(Predictor, self).__init__()

        # Want p(x_i | x_{not i}). x_i is the target and x_{not i} is the input trees

        # First "Fully Connected" Layer for each target tree and input tree. we will discard the nodes when target tree
        # is the same as input tree later
        first_linear_tensor = torch.Tensor(1, num_trees, num_trees, embedding_size, hidden_size).normal_(0, 0.01)
        self.linear1 = torch.nn.Parameter(first_linear_tensor)
        self.bias1 = torch.nn.Parameter(torch.Tensor(1, num_trees, num_trees, hidden_size).zero_())

        self.dropout = torch.nn.Dropout(p=0.5)

        # Second Fully Connected" layer for each target trees
        second_linear_tensor = torch.Tensor(1, num_trees, (num_trees - 1) * hidden_size, embedding_size).normal_(0, 0.01)
        self.linear2 = torch.nn.Parameter(second_linear_tensor)
        self.bias2 = torch.nn.Parameter(torch.Tensor(1, num_trees, embedding_size).zero_())

        self.hidden_size = hidden_size
        self.num_trees = num_trees

        # mask_tensor for discarding the nodes when target tree is the same as input tree.
        mask_tensor = ~torch.eye(num_trees).bool().unsqueeze(0).unsqueeze(-1)
        self.mask = torch.nn.Parameter(mask_tensor, requires_grad=False)

    def forward(self, x):
        # x: (bs, 1, n, e)

        # perform individual FC layers on the embedding dimension
        # x.unsqueeze(4) (bs, 1, n, e, 1)
        # linear1: (1, n, n, e, h)
        x = torch.sum(x.unsqueeze(4) * self.linear1, dim=3)
        # x: (bs, n, n, h),
        # bias1: (1, n, n, h)
        x = x + self.bias1

        # (bs, n, n, h)
        x = F.relu(x)
        x = self.dropout(x)

        # discard when the target is the same as the input.
        m = self.mask.expand([x.shape[0], self.num_trees, self.num_trees, self.hidden_size])
        # slice to (bs, n, n-1, h) and reshape (bs, n, (n-1)*h, 1)
        x = x.masked_select(m).view(x.shape[0], self.num_trees, (self.num_trees - 1) * self.hidden_size, 1)
        # x: (bs, n, (n-1)h, 1)

        # perform individual FC layers on the last dimension
        # linear2: (1, n, (n-1)h, e)
        x = torch.sum(x * self.linear2, dim=2)
        # x: (bs, n, e)
        # bias2: (1, n, e)
        x = x + self.bias2

        # (bs, n, e)
        return x


class XGBEmbedding(torch.nn.Module):
    def __init__(self, num_trees, num_nodes, embedding_size, hidden_size=10):
        super(XGBEmbedding, self).__init__()
        # All embeddings (flattened)
        self.emb = torch.nn.Embedding(num_nodes * num_trees, embedding_size)

        # See above
        self.predictors = Predictor(num_trees, embedding_size, hidden_size)

        # The "Training embedding" for getting the probability of each node
        self.linear = torch.nn.Parameter(torch.Tensor(1, num_trees, embedding_size, num_nodes).normal_(0, 0.01))

        self.num_trees = num_trees
        self.num_nodes = num_nodes

        # For use in "shift_index"
        self.arange = torch.nn.Parameter(torch.arange(self.num_trees), requires_grad=False)

    # This function shift the index for the flattened embedding.
    # For example, if sample x falls into the 0th node in the first tree and the 4th node in the second tree.
    # and each tree has 8 nodes (max), then sample x will have leaf index of 0 and 8 + 4 = 12.
    # This is needed because the embedding is flattened.
    def shift_index(self, x):
        # x: (bs, n)
        return x + (self.arange * self.num_nodes).unsqueeze(0)

    def forward(self, x):
        # Get the embedding for the leaf
        batch_emb = self.emb(self.shift_index(x))  # (bs, n, e)

        # just expand dims
        x_expanded = self.prepare(batch_emb)  # (bs, 1, n, e)

        # output the predictor of embedding
        net = self.predictors(x_expanded)  # (bs, n, e)

        # perform "dot product" with the "training embedding"
        net = torch.sum(net.unsqueeze(3) * self.linear, dim=2)

        # get the cross entropy per sample per tree.
        net = net.view(-1, net.shape[-1])
        loss = F.cross_entropy(net, x.view(-1), reduction="none")
        return loss.view(x.shape[0], self.num_trees)

    def prepare(self, x):
        # (bs, n, e) --> (bs, 1, n, e)
        x = x.unsqueeze(1)
        return x

    def get_weights(self):
        return self.emb.weight.view((self.num_trees, self.num_nodes, -1))

    def inference(self, x):
        # x: (bs, n)
        return self.emb(self.shift_index(x))  # (bs, n, e)
