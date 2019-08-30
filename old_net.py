import torch
import torch.nn.functional as F

class predictor(torch.nn.Module):
    def __init__(self, num_trees, embedding_size, hidden_size=2):
        super(predictor, self).__init__()

        self.linears = torch.nn.Parameter(torch.Tensor(num_trees - 1, embedding_size, hidden_size).normal_(0, 0.01))
        self.bias = torch.nn.Parameter(torch.Tensor(num_trees - 1, hidden_size).zero_())
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear((num_trees - 1) * hidden_size, embedding_size)

    def forward(self, x):
        # x: (bs, n-1, e)
        x = torch.einsum('abc,bcd->abd', x, self.linears)
        x = x + self.bias.unsqueeze(0)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view((-1, self.hidden_size * (self.num_trees - 1)))
        # return (bs, e)
        return self.fc(x)


class xgb_embedding(torch.nn.Module):
    def __init__(self, num_trees, num_nodes, embedding_size, hidden_size):
        super(xgb_embedding, self).__init__()
        #         self.emb = torch.nn.ModuleList([torch.nn.Embedding(num_nodes, embedding_size) for i in range(num_trees)])
        self.emb = torch.nn.Embedding(num_nodes * num_trees, embedding_size)
        self.predictors = torch.nn.ModuleList(
            [predictor(num_trees, embedding_size, hidden_size) for _ in range(num_trees)])
        self.num_trees = num_trees
        self.num_nodes = num_nodes

    # x : (bs, n)
    #     def forward(self, x):
    # #         batch_emb = self.emb(x) #(bs, n, e)
    #         total_loss = torch.zeros((x.shape[0]))
    #         for i in range(self.num_trees):
    #             arange = [j for j in range(self.num_trees) if j != i]
    #             net_branch = F.one_hot(x[:, arange], self.num_nodes)
    #             net_branch = net_branch.view((-1, (self.num_trees - 1) * self.num_nodes))
    #             net_branch = self.predictors[i](net_branch.float())
    #             net_branch = net_branch.unsqueeze(1) * self.emb[i].weight.unsqueeze(0)
    #             net_branch = torch.sum(net_branch, dim=-1) #(bs,m)
    #             target = x[:, i]
    #             loss = F.cross_entropy(net_branch, target, reduction="none")
    #             total_loss += loss
    #             print(i)
    #         return total_loss

    def shift_index(self, x):
        # x: (bs, n)
        return x + (torch.arange(self.num_trees) * self.num_nodes).unsqueeze(0)

    def forward(self, x):
        batch_emb = self.emb(self.shift_index(x))  # (bs, n, e)
        # raise Exception("GOOD")
        total_loss = torch.zeros((x.shape[0]))
        for i in range(self.num_trees):
            arange = [j for j in range(self.num_trees) if j != i]
            net_branch = self.predictors(batch_emb[:, arange, :])
            net_branch = net_branch.unsqueeze(1) * self.emb.weight[i*self.num_nodes:(i+1)*self.num_nodes, :].unsqueeze(0)
            net_branch = torch.sum(net_branch, dim=-1)  # (bs,m)
            target = x[:, i]
            loss = F.cross_entropy(net_branch, target, reduction="none")
            total_loss += loss
            print(i)
        return total_loss