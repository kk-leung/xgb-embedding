import pandas as pd
from sklearn.model_selection import train_test_split


class Splitter:
    def __init__(self, path, args):
        df = pd.read_csv(path)
        X, y = df.iloc[:, 2:], df['target']
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, random_state=args.random_state, train_size=0.6, test_size=0.4)
        Xvalid, Xtest, yvalid, ytest = train_test_split(Xvalid, yvalid, random_state=args.random_state, train_size=0.5, test_size=0.5)
        self.train = Xtrain, ytrain
        self.valid = Xvalid, yvalid
        self.test = Xtest, ytest