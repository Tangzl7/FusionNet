from yacs.config import CfgNode as CN

class Config(object):
    def __init__(self, config_yaml):
        self._C = CN()
        self._C.GPU = [0]

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002

        self._C.TRAINING = CN()
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.TRAIN_DIR = '../data'
        self._C.TRAINING.SAVE_DIR = '../checkpoints'

        self._C.merge_from_file(config_yaml)
        self._C.freeze()

    def dump(self, file_path):
        self._C.dump(stream=open(file_path, 'w'))
    
    def __getattr__(self, attr):
        return self._C.__getattr__(attr)
    
    def __repr__(self):
        return self._C.__repr__()
