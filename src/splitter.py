import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Splitter:
    def __init__(self, path, args):
        df = pd.read_csv(path)
        X, y = df.iloc[:, 2:], df['target']
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, random_state=args.random_state, train_size=0.6,
                                                          test_size=0.4)
        Xvalid, Xtest, yvalid, ytest = train_test_split(Xvalid, yvalid, random_state=args.random_state, train_size=0.5,
                                                        test_size=0.5)
        self.scaler = StandardScaler()
        Xtrain = self.scaler.fit_transform(Xtrain)
        Xvalid = self.scaler.transform(Xvalid)
        Xtest = self.scaler.transform(Xtest)
        self.train = Xtrain, ytrain.values
        self.valid = Xvalid, yvalid.values
        self.test = Xtest, ytest.values
        self.num_input = Xtrain.shape[1]
