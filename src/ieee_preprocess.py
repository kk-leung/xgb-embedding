import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.Timer import Timer


class IEEESplitter:
    def __init__(self, args):
        self.args = args
        self.id_cat_col = ["id_" + str(i) for i in range(12, 39)] + ["DeviceType",
                                                                     "DeviceInfo", "id_33_1", "id_31_1", "id_30_1"]
        self.trans_cat_col = ["M" + str(i) for i in range(1, 10)] + ["card" + str(i) for i in range(1, 7)] + \
                             ["ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
        self.try_list = ["id_" + str(i) for i in [13, 17, 18, 19, 20, 21, 22, 24, 25, 26]]

        self.id_onehot_encoder = OneHot(0.01)
        self.trans_onehot_encoder = OneHot(0.0025)
        self.id_scaler = StandardScaler()
        self.trans_scaler = StandardScaler()

        self.global_name = []
        self.global_df = []

        self.timer = Timer()
        df_trans = pd.read_csv("../data/ieee/train_transaction.csv")
        df_id_raw = pd.read_csv("../data/ieee/train_identity.csv")
        dt_trans = pd.read_csv("../data/ieee/test_transaction.csv")
        dt_id = pd.read_csv("../data/ieee/test_identity.csv")
        self.timer.toc("read done")

        self.test_id = dt_trans['TransactionID']


        df_trans, dv_trans = train_test_split(df_trans, random_state=args.random_state, train_size=0.8,
                                              test_size=0.2)
        train_ids = df_trans[['TransactionID']]
        df_id = train_ids.merge(df_id_raw, how='left', on='TransactionID')

        valid_ids = dv_trans[['TransactionID']]
        dv_id = valid_ids.merge(df_id_raw, how='left', on='TransactionID')
        self.timer.toc("split done")

        self.feature_engineering_id(df_id)
        self.feature_engineering_id(dv_id)
        self.feature_engineering_id(dt_id)
        self.timer.toc("process id done")

        self.global_count(df_id, df_trans)
        self.timer.toc("global_count done")

        df_trans.fillna(0, inplace=True)
        dt_trans.fillna(0, inplace=True)
        df_id_raw.fillna(0, inplace=True)
        dt_id.fillna(0, inplace=True)

        df_trans, ytrain = self.split_x_y(df_trans)
        dv_trans, yvalid = self.split_x_y(dv_trans)
        dt_trans, ytest = self.split_x_y(dt_trans)

        df_global = self.get_derived(df_id)
        self.timer.toc("get train derived done")
        dv_global = self.get_derived(dv_id)
        self.timer.toc("get valid derived done")
        dt_global = self.get_derived(dt_id)
        self.timer.toc("get test derived done")

        self.init_onehot(df_id, df_trans)
        self.init_scaler(df_id.drop(self.id_cat_col, axis=1), df_trans.drop(self.trans_cat_col, axis=1))

        Xtrain = self.transform(df_id, df_trans, df_global)
        self.timer.toc("transform train done")
        Xvalid = self.transform(dv_id, dv_trans, dv_global)
        self.timer.toc("transform valid done")
        Xtest = self.transform(dt_id, dt_trans, dt_global)
        self.timer.toc("transform test done")

        self.train = Xtrain, ytrain.values
        self.valid = Xvalid, yvalid.values
        self.test = Xtest, ytest.values
        self.num_input = Xtrain.shape[1]

    def transform(self, df_id, df_trans, df_global):
        id_num, id_cat = self.id_onehot_encoder.transform(df_id)
        id_num = self.id_scaler.transform(id_num.drop("TransactionID", axis=1))
        id = pd.concat([df_global, id_num, id_cat], axis=1)

        trans_num, trans_cat = self.trans_onehot_encoder.transform(df_trans)
        trans_num = self.trans_scaler.transform(trans_num.drop("TransactionID", axis=1))
        trans = pd.concat([df_trans[['TransactionID']], trans_num, trans_cat], axis=1)
        return trans.merge(id, how='left', on='TransactionID').drop("TransactionID", axis=1)

    def init_onehot(self, df_id, df_trans):
        self.id_onehot_encoder.fit(df_id, self.id_cat_col)
        self.trans_onehot_encoder.fit(df_trans, self.trans_cat_col)

    def init_scaler(self, df_id, df_trans):
        self.id_scaler.fit(df_id.drop("TransactionID", axis=1))
        self.trans_scaler.fit(df_trans.drop("TransactionID", axis=1))

    def get_derived(self, df_id):
        temp = df_id[self.try_list]
        df_list = [df_id[['TransactionID']]]
        for i in range(len(self.global_name)):
            li = self.global_name[i]
            column_name = self.global_df[i].columns[0]
            df_list.append(temp.merge(self.global_df[i].reset_index(), how='left', on=li)[[column_name]])
        return pd.concat(df_list, axis=1)

    @staticmethod
    def split_x_y(df):
        if 'isFraud' in df.columns:
            return df.drop('isFraud', axis=1), df['isFraud']
        else:
            y = pd.Series(np.zeros(len(df)))
            y.iloc[:10] = 1

            return df, y

    @staticmethod
    def parse_device_info(y):
        if type(y) is not str:
            return np.nan
        x = y.lower()
        if x.startswith("rv"):
            return "rv"
        elif x.startswith("sm-"):
            return "sm"
        elif x.startswith("trident"):
            return "trident"
        elif x.startswith("moto"):
            return "moto"
        elif x.startswith("lg"):
            return "lg"
        elif x.startswith("samsung"):
            return "sm"
        elif x.startswith("windows"):
            return "windows"
        elif x.startswith("ios"):
            return "ios"
        elif x.startswith("macos"):
            return "macos"
        else:
            return np.nan

    @staticmethod
    def feature_engineering_id(dg):
        dg["id_33_1"] = dg["id_33"].apply(lambda x: int(x.split("x")[0]) if type(x) is str else np.nan)
        dg["id_33"] = dg["id_33"].apply(lambda x: int(x.split("x")[1]) if type(x) is str else np.nan)

        dg["id_31_1"] = dg["id_31"].apply(lambda x: x.split(" ")[0] if type(x) is str else np.nan)
        dg["id_31"] = dg["id_31"].apply(
            lambda x: x.split(" ")[1] if type(x) is str and len(x.split(" ")) > 1 else np.nan)

        dg["id_30_1"] = dg["id_30"].apply(lambda x: x.split(" ")[0] if type(x) is str else np.nan)
        dg["id_30"] = dg["id_30"].apply(
            lambda x: x.split(" ")[1] if type(x) is str and len(x.split(" ")) > 1 else np.nan)

        dg["DeviceInfo"] = dg["DeviceInfo"].apply(IEEESplitter.parse_device_info)

    def global_count(self, df_id, df_trans):
        try_lists = IEEESplitter.powerset(self.try_list)
        df_temp = df_id.merge(df_trans[['TransactionID', 'isFraud']], how='left', on='TransactionID')
        df_temp = df_temp[['TransactionID', 'isFraud'] + self.try_list]
        for selected in try_lists:
            selected = list(selected)
            groupby = df_temp[['isFraud'] + selected].groupby(selected).mean()
            groupby.columns = ["global_" + "_".join(selected)]
            self.global_name.append(selected)
            self.global_df.append(groupby)

    @staticmethod
    def powerset(li):
        l = []
        from itertools import combinations
        for i in range(1, len(li) + 1):
            l = l + list(combinations(li, i))
        return l

    def export(self, inference, out_csv):
        df = pd.DataFrame()
        df['TransactionID'] = self.test_id
        df['isFraud'] = inference
        df.to_csv("../data/ieee/" + out_csv, index=False)


class OneHot:
    def __init__(self, col_selector_cutoff):
        self.catmap = {}
        self.cat_col = []
        self.col_selector_cutoff = col_selector_cutoff

    def fit(self, df, cat_col):
        self.catmap = {}
        self.cat_col = []
        for col in cat_col:
            self.cat_col.append(col)
            self.catmap[col] = []
            vc = df[col].value_counts()
            vc = vc[vc > len(df[col]) * self.col_selector_cutoff]
            for key in vc.keys():
                self.catmap[col].append(key)

    def one_hot(self, df):
        dg = pd.DataFrame()
        for col in self.cat_col:
            for key in self.catmap[col]:
                dg[col + " " + str(key)] = df[col].apply(lambda s: '1' if s == key else '')
        return dg

    def transform(self, df):
        df_one_hot = self.one_hot(df)
        dg = df.drop(self.cat_col, axis=1)
        # dg = pd.concat([dg, df_one_hot], axis=1)
        return dg, df_one_hot

    def fit_transform(self, df, cat_col):
        self.fit(df, cat_col)
        return self.transform(df)
