import numpy as np
import torch
import torch.nn.functional as F


class XGBEmbeddingEvaluator:
    def __init__(self, embedding, trees, print_eval=True):
        self.weight = torch.Tensor(embedding)
        self.trees = trees
        self.dot = self.dot_product()
        self.dot_norm = self.dot_product(normalize=True)
        if print_eval:
            self.eval_cover_corr()
            for _ in range(10):
                self.eval_nodes()

    def dot_product(self, normalize=False):
        if normalize:
            result = F.normalize(self.weight, p=2, dim=-1)
        else:
            result = self.weight

        # shape = (n, m, m)
        return torch.matmul(result, result.permute(0, 2, 1)).detach().cpu().numpy()

    def eval_cover_corr(self):
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
            cover.append(cover_corr)
            cover_norm.append(cover_corr_norm)
            value.append(value_corr)
            value_norm.append(value_corr_norm)
        print("")
        print(np.mean(cover))
        print(np.mean(cover_norm))

    def _print_path(self, tree_index, node_index):
        tree = self.trees[tree_index]
        leaf_node_index = list(tree.leaf_nodes.keys())[node_index]
        node = tree.leaf_nodes[leaf_node_index]
        string = ""
        while node.node_id in tree.parent_nodes.keys():
            parent_node_index = tree.parent_nodes[node.node_id]
            child_node_index = node.node_id
            node = tree.decision_nodes[parent_node_index]
            comparator_string = "<" if node.yes_node_id == child_node_index else ">"
            string = "{:5}{:2}{:6.2f}, ".format(node.feature_name, comparator_string,
                                                          node.decision_value) + string
        return "Tree {:5}, ".format(tree_index + 1) + string

    def eval_nodes(self, top_k=5, normalize=True):
        tree_index = np.random.randint(len(self.trees))
        num_nodes = len(self.trees[tree_index].leaf_nodes)
        node_index = np.random.randint(num_nodes)
        if normalize:
            all_weights = F.normalize(self.weight, p=2, dim=-1)
        else:
            all_weights = self.weight
        weight = all_weights[tree_index, node_index, :].unsqueeze(1)
        values = torch.matmul(all_weights, weight).detach().cpu().numpy()
        indexes = np.unravel_index(np.argsort(values, axis=None), dims=values.shape)
        indexes = list(zip(indexes[0][-top_k:][::-1], indexes[1][::-1]))
        print("Chosen Path:", self._print_path(tree_index, node_index))
        print("")
        i = 0
        while i < top_k:
            index = indexes[i]
            if index[1] < len(self.trees[index[0]].leaf_nodes):
                print("{:11}:".format(i + 1), self._print_path(index[0], index[1]))
                i += 1
        print("")
