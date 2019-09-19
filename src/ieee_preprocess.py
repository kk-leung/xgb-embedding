import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.Timer import Timer


class IEEESplitter:
    def __init__(self, args, load_raw=False):

        self.train_parsed_file = '../data/data/ieee/train.csv'
        self.valid_parsed_file = '../data/data/ieee/valid.csv'
        self.test_parsed_file = '../data/data/ieee/test.csv'
        self.args = args
        self.id_cat_col = ["id_" + str(i) for i in range(12, 39)] + ["DeviceType",
                                                                     "DeviceInfo", "id_33_1", "id_31_1", "id_30_1"]
        self.trans_cat_col = ["M" + str(i) for i in range(1, 10)] + ["card" + str(i) for i in range(1, 7)] + \
                             ["ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
        # self.try_list = ["id_" + str(i) for i in [13, 17]]
        self.try_list = ["id_" + str(i) for i in []]

        self.id_onehot_encoder = OneHot(0.01)
        self.trans_onehot_encoder = OneHot(0.01)
        self.id_scaler = Scaler()
        self.trans_scaler = Scaler()

        self.global_name = []
        self.global_df = []

        self.emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
                  'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other',
                  'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum',
                  'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
                  'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
                  'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft',
                  'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
                  'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other',
                  'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other',
                  'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
                  'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
                  'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft',
                  'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink',
                  'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo',
                  'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
                  'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

        self.drop_col = ['TransactionDT', 'V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102',
                    'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304',
                    'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119',
                    'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135',
                    'V320', 'V303', 'V297', 'V120']
        self.us_emails = ['gmail', 'net', 'edu']

        self.timer = Timer()

        if load_raw:
            df_trans = pd.read_csv("../data/data/ieee/train_transaction.csv").set_index("TransactionID")
            df_id_raw = pd.read_csv("../data/data/ieee/train_identity.csv").set_index("TransactionID")
            dt_trans = pd.read_csv("../data/data/ieee/test_transaction.csv").set_index("TransactionID")
            dt_id = pd.read_csv("../data/data/ieee/test_identity.csv").set_index("TransactionID")
            self.timer.toc("read done")

            self.test_id = list(dt_trans.index)

            df_trans, dv_trans = train_test_split(df_trans, random_state=args.random_state, train_size=0.8,
                                                  test_size=0.2)

            df_id = df_id_raw.loc[df_trans.index.intersection(df_id_raw.index), :]

            dv_id = df_id_raw.loc[dv_trans.index.intersection(df_id_raw.index), :]
            self.timer.toc("split done")

            self.feature_engineering_id(df_id)
            self.feature_engineering_id(dv_id)
            self.feature_engineering_id(dt_id)
            self.timer.toc("process id done")

            self.feature_engineering_trans(df_trans)
            self.feature_engineering_trans(dv_trans)
            self.feature_engineering_trans(dt_trans)
            self.timer.toc("process trans done")

            self.global_count(df_id, df_trans)
            self.timer.toc("global_count done")

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

            pd.concat([Xtrain, ytrain], axis=1).to_csv(self.train_parsed_file, float_format='%.8f', index=True)
            self.timer.toc("write train done")
            pd.concat([Xvalid, yvalid], axis=1).to_csv(self.valid_parsed_file, float_format='%.8f', index=True)
            self.timer.toc("write valid done")
            Xtest.to_csv(self.test_parsed_file, float_format='%.8f', index=True)
            self.timer.toc("write test done")

            raise Exception("STOP HERE!")
        else:
            self.train = self.load(self.train_parsed_file)
            self.timer.toc("load train done")
            self.valid = self.load(self.valid_parsed_file)
            self.timer.toc("load valid done")
            self.test = self.load(self.test_parsed_file, test=True)
            self.timer.toc("load test done")
            self.num_input = self.train[0].shape[1]

            # sanity check
            print(np.unique(self.train[1], return_counts=True))
            print(np.unique(self.valid[1], return_counts=True))

        # self.train = Xtrain, ytrain.values
        # self.valid = Xvalid, yvalid.values
        # self.test = Xtest, ytest.values
        # self.num_input = Xtrain.shape[1]

    def load(self, file, test=False):
        df = pd.read_csv(file)
        if not test:
            X = df.drop(['TransactionID', 'isFraud'], axis=1)
            y = df['isFraud']
        else:
            X = df.drop('TransactionID', axis=1)
            y = pd.Series(np.zeros(len(df)))
            y.iloc[:10] = 1
            self.test_id = df['TransactionID']
        return X.values, y.values


    def transform(self, df_id, df_trans, df_global):
        id_num, id_cat = self.id_onehot_encoder.transform(df_id)
        id_num = self.id_scaler.transform(id_num)
        id = pd.concat([id_num, id_cat], axis=1)
        # id = pd.concat([df_global, id_num, id_cat], axis=1)

        trans_num, trans_cat = self.trans_onehot_encoder.transform(df_trans)
        trans_num = self.trans_scaler.transform(trans_num)
        trans = pd.concat([trans_num, trans_cat], axis=1)

        return trans.merge(id, how='left', left_index=True, right_index=True)

    def init_onehot(self, df_id, df_trans):
        self.id_onehot_encoder.fit(df_id, self.id_cat_col)
        self.trans_onehot_encoder.fit(df_trans, self.trans_cat_col)

    def init_scaler(self, df_id, df_trans):
        self.id_scaler.fit(df_id)
        self.trans_scaler.fit(df_trans)

    def get_derived(self, df_id):
        return
        temp = df_id[self.try_list]
        df_list = [] # [df_id[['TransactionID']]]
        for i in range(len(self.global_name)):
            li = self.global_name[i]
            column_name = self.global_df[i].columns[0]
            merged = temp.merge(self.global_df[i].reset_index(), how='left', on=li)
            merged.index = temp.index
            df_list.append(merged[[column_name]])
        return pd.concat(df_list, axis=1)

    @staticmethod
    def split_x_y(df):
        if 'isFraud' in df.columns:
            return df.drop('isFraud', axis=1), df['isFraud']
        else:
            y = pd.Series(np.zeros(len(df)), index=df.index)
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

        dg["nulls_id"] = dg.isna().sum(axis=1)

    def feature_engineering_trans(self, dg):
        def get_suffix(x):
            suffix = str(x).split(".")[-1]
            return x if str(x) not in self.us_emails else 'us'

        dg["nulls_trans"] = dg.isna().sum(axis=1)
        for col in ["P_emaildomain", "R_emaildomain"]:
            dg[col + "_bin"] = dg[col].map(self.emails)
            dg[col + "_suffix"] = dg[col].apply(get_suffix)

        dg.drop(self.drop_col, axis=1, inplace=True)




    def global_count(self, df_id, df_trans):
        try_lists = IEEESplitter.powerset(self.try_list)
        df_temp = df_id.merge(df_trans[['isFraud']], how='left', left_index=True, right_index=True)
        df_temp = df_temp[['isFraud'] + self.try_list]
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
        df.to_csv("../data/data/ieee/" + out_csv, index=False)


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


class Scaler:
    def __init__(self):
        self.num_features = 0
        self.mean = None
        self.std = None

    def fit(self, df):
        self.num_features = len(df.columns)
        self.mean = np.nanmean(df.values, axis=0, keepdims=True)
        self.std = np.nanstd(df.values, axis=0, keepdims=True)

    def transform(self, df):
        assert self.num_features == len(df.columns)
        assert self.mean is not None
        assert self.std is not None
        values = df.values
        new_values = (values - self.mean) / self.std
        return pd.DataFrame(new_values, index=df.index, columns=df.columns)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)



