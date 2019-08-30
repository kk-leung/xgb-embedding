import torch
import torch.nn.functional as F
import numpy as np

from xgb_dump_parser import decision_tree


class xgb_emb_eval:
    def __init__(self, test, model, trees):
        self.test = test
        self.weight = model.get_weights()
        self.trees = trees
        self.inference_result = self.inference(test, model)
        self.dot = self.dot_product()
        self.dot_norm = self.dot_product(normalize=True)

    def inference(self, test, model):
        model.cuda()
        model.eval()
        results = []
        for batch, x in enumerate(test):
            x = x.cuda()
            results.append(model.inference(x))
        return torch.cat(results, dim=0).detach().cpu().numpy()

    def dot_product(self, normalize=False):
        if normalize:
            result = F.normalize(self.weight, p=2, dim=-1)
        else:
            result = self.weight

        # shape = (n, m, m)
        return torch.matmul(result, result.permute(0, 2, 1)).detach().cpu().numpy()

    def eval(self):
        # dot = self.dot.reshape(len(self.trees), -1)
        # dot_norm = self.dot.reshape(len(self.trees), -1)
        cover = []
        value = []
        cover_norm = []
        value_norm = []
        for i in range(len(self.trees)):
            tree = self.trees[i]
            num_nodes = len(tree.leaf_nodes)
            iu = np.triu_indices(num_nodes, 1)
            cover_corr = np.corrcoef(tree.self_cover[iu], self.dot[i, :num_nodes, :num_nodes][iu])[0, 1]
            value_corr = np.corrcoef(tree.self_value[iu], self.dot[i, :num_nodes, :num_nodes][iu])[0, 1]
            cover_corr_norm = np.corrcoef(tree.self_cover[iu], self.dot_norm[i, :num_nodes, :num_nodes][iu])[0, 1]
            value_corr_norm = np.corrcoef(tree.self_value[iu], self.dot_norm[i, :num_nodes, :num_nodes][iu])[0, 1]
            if i < 20:
                print(i, cover_corr)
                print(i, cover_corr_norm)
                print(i, value_corr)
                print(i, value_corr_norm)
            cover.append(cover_corr)
            cover_norm.append(cover_corr_norm)
            value.append(value_corr)
            value_norm.append(value_corr_norm)
        print(cover)
        print(cover_norm)
        print(value)
        print(value_norm)
        print("")
        print(np.mean(cover))
        print(np.mean(cover_norm))
        print(np.mean(value))
        print(np.mean(value_norm))












