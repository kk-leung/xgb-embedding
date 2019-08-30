import torch
import torch.nn.functional as F

class predictor(torch.nn.Module):
    def __init__(self, num_trees, embedding_size, hidden_size=2):
        super(predictor, self).__init__()

        self.linear1 = torch.nn.Parameter(torch.Tensor(1, num_trees, num_trees, embedding_size, hidden_size).normal_(0, 0.01))
        self.bias1 = torch.nn.Parameter(torch.Tensor(1, num_trees, num_trees, hidden_size).zero_())
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Parameter(torch.Tensor(1, num_trees, (num_trees - 1) * hidden_size, embedding_size).normal_(0, 0.01))
        self.bias2 = torch.nn.Parameter(torch.Tensor(1, num_trees, embedding_size).zero_())
        self.hidden_size = hidden_size
        self.num_trees = num_trees

        self.mask = torch.nn.Parameter(~torch\
            .eye(num_trees)\
            .bool()\
            .unsqueeze(0)\
            .unsqueeze(-1), requires_grad=False)

    def forward(self, x):
        # x: (bs, 1, n, e)
        x = torch.sum(x.unsqueeze(4) * self.linear1, dim=3)
        x = x + self.bias1
        # (bs, n, n, h)
        x = F.relu(x)
        x = self.dropout(x)
        m = self.mask.expand([x.shape[0], self.num_trees, self.num_trees, self.hidden_size])
        #slice to (bs, n, n-1, h) and reshape (bs, n, (n-1)*h, 1)
        x = x.masked_select(m).view(x.shape[0], self.num_trees, (self.num_trees - 1) * self.hidden_size, 1)
        # (bs, n, (n-1)h, 1)
        x = torch.sum(x * self.linear2, dim=2)
        x = x + self.bias2
        # (bs, n, e)
        return x


class xgb_embedding(torch.nn.Module):
    def __init__(self, num_trees, num_nodes, embedding_size, hidden_size=10):
        super(xgb_embedding, self).__init__()
        self.emb = torch.nn.Embedding(num_nodes * num_trees, embedding_size)
        self.predictors = predictor(num_trees, embedding_size, hidden_size)
        self.linear = torch.nn.Parameter(torch.Tensor(1, num_trees, embedding_size, num_nodes).normal_(0, 0.01))
        self.bias = torch.nn.Parameter(torch.Tensor(1, num_trees, num_nodes).zero_())
        self.num_trees = num_trees
        self.num_nodes = num_nodes
        self.arange = torch.nn.Parameter(torch.arange(self.num_trees), requires_grad=False)

    def shift_index(self, x):
        # x: (bs, n)
        return x + (self.arange * self.num_nodes).unsqueeze(0)

    def forward(self, x):
        batch_emb = self.emb(self.shift_index(x))  # (bs, n, e)
        x_expanded = self.prepare(batch_emb)
        net = self.predictors(x_expanded)
        net = torch.sum(net.unsqueeze(3) * self.linear, dim=2)
        net = net + self.bias
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

