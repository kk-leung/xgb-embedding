from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle

from src.Timer import Timer
from src.xgbembeddingevaluator import XGBEmbeddingEvaluator


class XgbSvdTrainer:
    def __init__(self, args, num_nodes, timer: Timer = None):
        self.timer = Timer() if timer is None else timer
        self.args = args
        self.num_nodes = num_nodes
        self.cumsum = np.cumsum([0] + self.num_nodes)
        self.embedding_size = args.embedding_size
        self.embeddings = None
        if args.load:
            model_name = "{:s}_{:d}_{:d}".format(self.args.svd_name, self.args.max_depth,
                                                 self.args.num_trees_for_embedding)
            with open(model_name, "rb") as f:
                self.svds = pickle.load(f)
                self._get_weights()
        else:
            self.svds = TruncatedSVD(n_components=self.embedding_size)

    def fit_svd(self, x):
        if not self.args.load:
            self.svds.fit(x)
            model_name = "{:s}_{:d}_{:d}".format(self.args.svd_name, self.args.max_depth,
                                                 self.args.num_trees_for_embedding)
            self._get_weights()
            with open(model_name, "wb") as f:
                pickle.dump(self.svds, f)

    def _get_weights(self):
        max_length = max(self.num_nodes)
        embeddings = np.zeros((len(self.num_nodes), max_length, self.embedding_size))
        for i in range(len(self.num_nodes)):
            embeddings[i, :self.num_nodes[i], :] = \
                np.transpose(self.svds.components_)[self.cumsum[i]:self.cumsum[i+1], :]
        self.embeddings = embeddings
        return embeddings  # (n_trees, n_nodes, e)

    def inference(self, x):
        # (bs, n_trees, e)
        return np.stack([self.embeddings[i][x[:, i]] for i in range(len(self.embeddings))], axis=1)

    # def get_embedding(self, x_list, trees):
    #     XGBEmbeddingEvaluator(self.embeddings, trees, print_eval=self.args.print_eval)
    #     return [self.inference(x) for x in x_list]
