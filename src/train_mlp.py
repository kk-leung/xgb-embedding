import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import Dataset, DataLoader

from src.mlp_net import MLP
from src.splitter import Splitter
from src.xgbtrainer import timer


class MLPTrainer:
    def __init__(self, args, num_input, mode='both'):
        self.args = args

        if mode is 'raw_only':
            num_features = num_input
        elif mode is 'emb_only':
            num_features = args.embedding_size * args.num_round
            # num_features = args.embedding_size
        elif mode is 'both':
            num_features = args.embedding_size * args.num_round + num_input
            # num_features = args.embedding_size + num_input
        else:
            raise Exception("unidentified mode. possible={'raw_only', 'emb_only', 'both'}")

        self.model = MLP(args, num_features)
        self.mode = mode
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        total_params = sum(x.data.nelement() for x in self.model.parameters())
        print("Model total number of parameters: {}".format(total_params))

    def trainIters(self, train, valid):
        tim = timer()

        # ADAM opts
        opt = optim.Adam(self.model.parameters(), lr=self.args.mlp_lr, weight_decay=self.args.mlp_weight_decay)

        ################ Training epoch
        self.model.cuda()
        for epoch in range(1, self.args.mlp_num_epoch + 1):
            self.model.train()
            train_losses = self.train_model(opt, train)

            valid_losses, results, ground_truths = self.valid_model(valid)
            valid_auc = self.evaluate(results, ground_truths)

            tim.toc("epoch {:4d} - train loss: {:10.6f}   valid loss: {:10.6f}   valid auc: {:10.6f}".format(epoch, np.mean(train_losses),
                                                                                       np.mean(valid_losses), valid_auc))


    def valid_model(self, valid):
        valid_losses = []
        results = []
        ground_truths = []

        self.model.eval()
        for batch, x in enumerate(valid):
            out = self.model(x[0].cuda())
            loss = self.loss(out, x[1].cuda())
            valid_losses.append(loss.item())
            results.append(torch.sigmoid(out))
            ground_truths.append(x[1])
            # x = x.cpu()
        inference_result = torch.cat(results, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()

        return valid_losses, inference_result, ground_truth

    def train_model(self, opt, train):
        train_losses = []
        for batch, x in enumerate(train):
            opt.zero_grad()
            out = self.model(x[0].cuda())
            loss = self.loss(out, x[1].cuda())
            loss.backward()
            opt.step()
            # x = x.cpu()
            train_losses.append(loss.item())
        return train_losses

    def inference(self, test):
        self.model.cuda()
        self.model.eval()
        results = []
        ground_truths = []
        for batch, x in enumerate(test):
            results.append(torch.sigmoid(self.model(x[0].cuda())))
            ground_truths.append(x[1])
        return torch.cat(results, dim=0).detach().cpu().numpy(), torch.cat(ground_truths, dim=0).detach().cpu().numpy()

    @staticmethod
    def evaluate(pred, ground_truth, print_result=True):
        auc = roc_auc_score(ground_truth, pred)
        if print_result:
            print("AUC = ", auc)
            print("error = ",  1 - np.sum(np.round(pred) == ground_truth) / len(pred))
        return auc

    def get_loader(self, raw, emb, shuffle=True):
        X, y = raw
        if self.mode is 'both':
            norm = np.linalg.norm(emb, axis=-1, keepdims=True)
            emb = emb / norm
            emb = emb.reshape(-1, self.args.embedding_size * self.args.num_round)
            # emb = np.mean(emb, axis=1)
            X = np.concatenate([X, emb], axis=1)
        elif self.mode is 'raw_only':
            pass
        elif self.mode is 'emb_only':
            norm = np.linalg.norm(emb, axis=-1, keepdims=True)
            emb = emb / norm
            X = emb.reshape(-1, self.args.embedding_size * self.args.num_round)
            # X = np.mean(emb, axis=1)
        else:
            raise Exception("unidentified mode. possible={'raw_only', 'emb_only', 'both'}")
        dataset = MLPDataset(X, y)
        return DataLoader(dataset, batch_size=self.args.mlp_batch_size, shuffle=shuffle)

    def run(self, raws, embs):
        train_loader = self.get_loader(raws[0], embs[0])
        valid_loader = self.get_loader(raws[1], embs[1])
        test_loader = self.get_loader(raws[2], embs[2], shuffle=False)
        self.trainIters(train_loader, valid_loader)
        train_pred, train_true = self.inference(train_loader)
        valid_pred, valid_true = self.inference(valid_loader)
        test_pred, test_true = self.inference(test_loader)
        print('train')
        self.evaluate(train_pred, train_true)
        print('valid')
        self.evaluate(valid_pred, valid_true)
        print('test')
        self.evaluate(test_pred, test_true)


class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return torch.FloatTensor(self.X[index, :]), torch.FloatTensor(self.y[index:index + 1])



