import os
import cv2
import sys
import pdb
import time
import argparse
import numpy as np

from fusion_net import FusionNet
from dataset import DataLoaderForFN
from losses import DataLoss, SmoothingLoss, EdgeLoss

import torch
import torch.optim
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.autograd as autograd

def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	net = FusionNet().cuda()
	# net.load_state_dict(torch.load('./snapshots/denoisy_.pth'))

	# net.apply(weights_init)
	if config.load_pretrain == True:
		net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = DataLoaderForFN(config.images_path)
	print(len(train_dataset))
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)


	data_loss = DataLoss()
	edge_loss = EdgeLoss()
	smoothing_loss = SmoothingLoss()

	optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	net.train()

	for epoch in range(config.num_epochs):
		for iteration, data in enumerate(train_loader):

			l, nir, gt, nir_mask = data[0], data[1], data[2], data[3]
			l, nir, gt, nir_mask = l.cuda(), nir.cuda(), gt.cuda(), nir_mask.cuda()
			# pdb.set_trace()

			denoise, out = net(l, nir, nir_mask)

			loss_edge = 10*edge_loss(out, gt, nir_mask, 1)
			loss_data = data_loss(out, gt)
			loss = loss_edge + loss_data

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(net.parameters(),config.grad_clip_norm)
			optimizer.step()

			print("epoch", epoch, "Loss at iteration", iteration+1, ":", loss_edge.item(), loss_data.item())
			out = torch.squeeze(out, 0).cpu().detach().numpy()
			out = np.transpose(out, (1, 2, 0)) * 255.
			cv2.imwrite('fusion_tmp.png', np.uint8(out))
	
	torch.save(net.state_dict(), config.snapshots_folder + 'fusion_cnn.pth') 




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--images_path', type=str, default="../data")
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=12)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--val_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--display_iter', type=int, default=5)
	parser.add_argument('--snapshot_iter', type=int, default=5)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/joint_filter_cnn.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)
