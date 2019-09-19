import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import Dataset, DataLoader

from src.earlystopper import EarlyStopper
from src.Timer import Timer
from src.mlp_net import MLP
from src.santandersplitter import SantanderSplitter



class MLPTrainer:
    def __init__(self, args, num_input, mode='both', timer: Timer = None):
        self.args = args
        self.timer = Timer() if timer is None else timer

        if args.all_trees and False:
            self.embedding_size_to_mlp = args.embedding_size # * args.num_trees_for_embedd
        else:
            self.embedding_size_to_mlp = args.embedding_size * args.num_trees_for_embedding
        if mode is 'raw_only':
            num_features = num_input
        elif mode is 'emb_only':
            num_features = self.embedding_size_to_mlp
            # num_features = args.embedding_size
        elif mode is 'both':
            num_features = self.embedding_size_to_mlp + num_input
            # num_features = args.embedding_size + num_input
        else:
            raise Exception("unidentified mode. possible={'raw_only', 'emb_only', 'both'}")

        if args.load_mlp:
            model_name = "{:s}_{:d}_{:d}.chkpt".format(self.args.mlp_model_name, self.args.max_depth,
                                                       self.args.num_trees_for_embedding)
            self.model = torch.load(model_name)['model']
        else:
            self.model = MLP(args, num_features)
        self.mode = mode
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        total_params = sum(x.data.nelement() for x in self.model.parameters())
        print("Model total number of parameters: {}".format(total_params))

    def trainIters(self, train, valid):

        early_stopper = EarlyStopper(3, 'moving', reverse=True)
        max_auc = 0

        # ADAM opts
        opt = optim.Adam(self.model.parameters(), lr=self.args.mlp_lr, weight_decay=self.args.mlp_weight_decay)

        ################ Training epoch
        self.model.cuda()
        for epoch in range(1, self.args.mlp_num_epoch + 1):
            self.model.train()
            train_losses = self.train_model(opt, train)

            valid_losses, results, ground_truths = self.valid_model(valid)
            valid_auc = self.evaluate(results, ground_truths, print_result=False)

            self.timer.toc("epoch {:4d} - train loss: {:10.6f}   valid loss: {:10.6f}   valid auc: {:10.6f}".format(epoch, np.mean(train_losses),
                                                                                       np.mean(valid_losses), valid_auc))

            checkpoint = {'model': self.model, 'args': self.args}
            model_name = "{:s}_{:d}_{:d}.chkpt".format(self.args.mlp_model_name, self.args.max_depth,
                                                       self.args.num_trees_for_embedding)

            if valid_auc > max_auc:
                max_auc = valid_auc
                torch.save(checkpoint, model_name)

            if early_stopper.record(valid_auc):
                self.model = torch.load(model_name)['model']
                return

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
            results.append(torch.sigmoid(self.model(x[0].cuda())).detach().cpu().numpy())
            ground_truths.append(x[1].detach().cpu().numpy())
        return np.concatenate(results, axis=0), np.concatenate(ground_truths, axis=0)
        # return torch.cat(results, dim=0).detach().cpu().numpy(), torch.cat(ground_truths, dim=0).detach().cpu().numpy()

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
            emb = emb / (norm + 1e-8)
            # if not self.args.all_trees:
            emb = emb.reshape(-1, self.embedding_size_to_mlp)
            # emb = np.mean(emb, axis=1)
            X = np.concatenate([X, emb], axis=1)
        elif self.mode is 'raw_only':
            pass
        elif self.mode is 'emb_only':
            norm = np.linalg.norm(emb, axis=-1, keepdims=True)
            emb = emb / (norm + 1e-8)
            # if not self.args.all_trees:
            X = emb.reshape(-1, self.embedding_size_to_mlp)
            # else:
            #     X = emb
            # X = np.mean(emb, axis=1)
        else:
            raise Exception("unidentified mode. possible={'raw_only', 'emb_only', 'both'}")
        dataset = MLPDataset(np.nan_to_num(X, copy=False), y)
        return DataLoader(dataset, batch_size=self.args.mlp_batch_size, shuffle=shuffle)

    def run(self, raws, embs):
        train_loader = self.get_loader(raws[0], embs[0])
        self.timer.toc("train loader done")
        valid_loader = self.get_loader(raws[1], embs[1])
        self.timer.toc("valid loader done")
        test_loader = self.get_loader(raws[2], embs[2], shuffle=False)
        self.timer.toc("test loader done")
        if not self.args.load_mlp:
            self.trainIters(train_loader, valid_loader)
        train_pred, train_true = self.inference(train_loader)
        self.timer.toc("train inference done")
        valid_pred, valid_true = self.inference(valid_loader)
        self.timer.toc("valid inference done")
        test_pred, test_true = self.inference(test_loader)
        self.timer.toc("test inference done")
        print('train')
        self.evaluate(train_pred, train_true)
        print('valid')
        self.evaluate(valid_pred, valid_true)
        print('test')
        self.evaluate(test_pred, test_true)
        return test_pred


class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return torch.FloatTensor(self.X[index, :]), torch.FloatTensor(self.y[index:index + 1])



