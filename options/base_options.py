import os.path as osp
import os


class parser(object):
    def __init__(self):
        self.name = "training_cloth_segm_u2net_exp1"  # Expriment name
        self.image_folder = "/content/drive/MyDrive/AIEngineer/data2019/datatrain2019"  # image folder path
        self.df_path = "/content/drive/MyDrive/AIEngineer/data2019/traindata2019.csv"  # label csv path
        self.distributed = False  # True for multi gpu training
        self.isTrain = True

        self.fine_width = 512
        self.fine_height = 512

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 4  # 12
        self.nThreads = 2  # 3
        self.max_dataset_size = float("inf")

        self.serial_batches = False
        self.continue_train = False
        if self.continue_train:
            self.unet_checkpoint = "prev_checkpoints/cloth_segm_unet_surgery.pth"

        self.save_freq = 1000
        self.print_freq = 200
        self.image_log_freq = 500

        self.iter = 4000
        self.lr = 0.0003
        self.clip_grad = 5

        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)
