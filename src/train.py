from statistics import mode
from config import Config
opt = Config('config.yml')

import os
import pdb
import time
import random
import numpy as np
from tqdm import tqdm

import losses
from model import FusionNet
from dataset import DataLoaderTrain
from warmup_scheduler import GradualWarmupScheduler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

train_dir = opt.TRAINING.TRAIN_DIR
model_dir = opt.TRAINING.SAVE_DIR

model = FusionNet()
model.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!\n\n")

lr = opt.OPTIM.LR_INITIAL
start_epoch = 1

######### Optimizer ###########
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Loss ###########
data_loss = losses.DataLoss()
edge_loss = losses.AreaEdgeLoss()

######### DataLoader ###########
train_dataset = DataLoaderTrain(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    model.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero grad
        for param in model.parameters():
            param.grad = None
        # pdb.set_trace()
        rgb, nir = data[0].cuda(), data[1].cuda()
        high_reflection_rgb, high_reflection_nir = data[2].cuda(), data[3].cuda()

        out = model(rgb, nir)
        loss = data_loss(out, rgb) + edge_loss(out, rgb, nir, high_reflection_rgb, high_reflection_nir)

        loss.backward()
        optimizer.step()
        epoch_loss += loss

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth")) 
