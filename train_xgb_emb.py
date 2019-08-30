import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch import optim
from time import time
import torch.nn.functional as F
from xgb_trainer import xgb_trainer, timer
from new_net import xgb_embedding
from xgb_emb_eval import xgb_emb_eval

# %% Args setup
parser = argparse.ArgumentParser()

################## Model params
parser.add_argument('--num_epoch', type=int, default=35)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model_name', type=str, default='xgb-emb')
parser.add_argument('--embedding_size', type=int, default=20)
parser.add_argument('--random_state', type=int, default=0)

parser.add_argument('--eta', type=float, default=0.2)
parser.add_argument('--max_depth', type=int, default=6)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)


#################### Finalizing args
args = parser.parse_args()
if args.random_state is not None:
    torch.manual_seed(args.random_state)
print("args = ", args)


def trainIters(args, train, valid, model):
    tim = timer()

    # ADAM opts
    opt = optim.Adam(model.parameters(), lr=args.lr)

    ################ Training epoch
    model.cuda()
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        train_losses = []
        for batch, x in enumerate(train):
            opt.zero_grad()
            x = x.cuda()
            loss = torch.mean(model(x))
            loss.backward()
            opt.step()
            # x = x.cpu()
            train_losses.append(loss.item())

        valid_losses = []
        model.eval()
        for batch, x in enumerate(valid):
            x = x.cuda()
            loss = torch.mean(model(x))
            valid_losses.append(loss.item())
            # x = x.cpu()
        tim.toc("epoch {:4d} - train loss: {:10.6f}   valid loss: {:10.6f}".format(epoch, np.mean(train_losses), np.mean(valid_losses)))

        checkpoint = {'model': model, 'args': args}
        model_name = args.model_name + '.chkpt'
        torch.save(checkpoint, model_name)

def inference(test, model):
    model.cuda()
    model.eval()
    results = []
    for batch, x in enumerate(test):
        x = x.cuda()
        results.append(model.inference(x))
    return torch.cat(results, dim=0).detach().cpu().numpy()

# %% Training
def main():

    trainer = xgb_trainer("../data/santander/train.csv", args)
    train, valid, test, test2 = trainer.get_loaders()
    model = xgb_embedding(trainer.num_trees, trainer.max_length, args.embedding_size)
    total_params = sum(x.data.nelement() for x in model.parameters())
    print("Model total number of parameters: {}".format(total_params))

    # training
    trainIters(args, train, valid, model)

    # eval
    xgb_eval = xgb_emb_eval(test, model, trainer.trees)
    xgb_eval.eval()


# %%
if __name__ == "__main__":
    main()