import os
from loguru import logger
import glob

class Exp():
    def __init__(self):

        # --------------------------------------------------------
        # config
        #
        # 引数情報
        self.symbol      = 'XLM-USD'          # 通貨レート
        self.granularity = 300              # 何秒置きにデータを取得するか(60, 300, 900, 3600, 21600, 86400) が指定可能
        self.start_date  = '2020-01-01-00-00' # データ取得範囲：開始日
        self.end_date    = '2020-01-29-00-00' # データ取得範囲：終了日

        self.dataset_name = "./data/datasets/v1.0/symbol-{}_granularity-{}_start-{}_end-{}.csv".format(self.symbol, self.granularity, self.start_date, self.end_date)
        self.dataset_train_name = "./data/datasets/v1.0/symbol-{}_granularity-{}_start-{}_end-{}_train.csv".format(self.symbol, self.granularity, self.start_date, self.end_date)
        self.dataset_valid_name = "./data/datasets/v1.0/symbol-{}_granularity-{}_start-{}_end-{}_valid.csv".format(self.symbol, self.granularity, self.start_date, self.end_date)

        # ログフォルダの生成
        self.log_dir = './logs/'
        self.datasets_dir = '../datasets/'
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

        # train data idx
        self.idx1 = 100
        self.idx2 = 5000

        # test data idx
        self.idx3 = 6000

        self.window_size = 100
        self.trade_fee = 0
        self.env_num = 20

        self.total_timesteps=100000

        # tb_log_name = "PPO2_feat19"
        # self.tb_log_name = "PPO2_feat57_100_lstm128"
        self.tb_log_name = "PPO2_no_feat_sample_v1"

        DATASET_GET_FLAG = False

        self.n_steps = 128
        self.save_freq = 1000
        self.eval_freq = 10000
        
        # ---------
        # load save model
        #
        model_list = sorted(glob.glob("logs/*.zip"))
        
        if(len(model_list)>0):
            self.model_latest = model_list[-1].split("/")[-1].split(".zip")[0].split("_step")[0]
        else:
            self.model_latest = None
            
        if(self.model_latest):
            self.resume_FLAG = True
            self.model_name, self.resume_idx, self.train_num = self.model_latest.split("_")
        else:
            self.resume_FLAG = False
            self.resume_idx    = 0
            self.train_num     = 1

        

        
        
    def preview_param(self):
        for k, v in vars(self).items():
            logger.info("{:>30}:{}".format(k, v))
        return vars(self)
    