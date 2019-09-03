import numpy as np
import torch
from torch import optim

from src.mlp_net import MLP
from src.xgbembeddingevaluator import XGBEmbeddingEvaluator
from src.xgbtrainer import timer


class MLPTrainer:
    def __init__(self, args, mode='both'):
        if mode is 'raw_only':
            pass
        elif mode is 'emb_only':
            pass
        elif mode is 'both':
            pass
        else:
            pass
        self.args = args
        self.model = MLP(args, )
        total_params = sum(x.data.nelement() for x in self.model.parameters())
        print("Model total number of parameters: {}".format(total_params))

    def trainIters(self, train, valid):
        tim = timer()

        # ADAM opts
        opt = optim.Adam(self.model.parameters(), lr=self.args.lr)

        ################ Training epoch
        self.model.cuda()
        for epoch in range(1, self.args.num_epoch + 1):
            self.model.train()
            train_losses = self.train_model(opt, train)

            valid_losses = self.valid_model(valid)
            tim.toc("epoch {:4d} - train loss: {:10.6f}   valid loss: {:10.6f}".format(epoch, np.mean(train_losses),
                                                                                       np.mean(valid_losses)))

            checkpoint = {'model': self.model, 'args': self.args}
            model_name = self.args.model_name + '.chkpt'
            torch.save(checkpoint, model_name)

    def valid_model(self, valid):
        valid_losses = []
        self.model.eval()
        for batch, x in enumerate(valid):
            x[0] = x[0].cuda()
            loss = torch.mean(self.model(x[0]))
            valid_losses.append(loss.item())
            # x = x.cpu()
        return valid_losses

    def train_model(self, opt, train):
        train_losses = []
        for batch, x in enumerate(train):
            opt.zero_grad()
            x[0] = x[0].cuda()
            loss = torch.mean(self.model(x[0]))
            loss.backward()
            opt.step()
            # x = x.cpu()
            train_losses.append(loss.item())
        return train_losses

    def inference(self, test):
        self.model.cuda()
        self.model.eval()
        results = []
        for batch, x in enumerate(test):
            x[0] = x[0].cuda()
            results.append(self.model.inference(x[0]))
        return torch.cat(results, dim=0).detach().cpu().numpy()

    def init_model(self, train, valid):
        if self.args.load is False:
            self.trainIters(train, valid)
        else:
            self.model.load(self.args.model_name + '.chkpt')
            self.valid_model(valid)

    def get_embedding(self, loaders, trees):
        XGBEmbeddingEvaluator(self.model, trees)
        emb = []
        for loader in loaders:
            emb.append(XGBEmbeddingEvaluator.inference_model(loader, self.model))
        return emb
