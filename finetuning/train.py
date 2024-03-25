import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import time
import argparse
from pathlib import Path

from datasets import *
import sys
sys.path.append("../")
from utils.setup_funcs import init_logger, init_seeds
from utils.setup_funcs import PROJECT_ROOT

from robustbench.utils import clean_accuracy
from robustbench.utils import load_model

np.set_printoptions(linewidth=np.nan)


def train(model,loss_fn,optimizer,log_name,epochs,ese,device,
		  train_loader,val_loader,logger,lr_scheduler,log_freq):

	# start tensorboard session
	writer = SummaryWriter(os.path.join(PROJECT_ROOT,"saved_data/runs",log_name)+"_"+str(time.time()))

	# log training parameters
	print("===========================================")
	for k,v in zip(locals().keys(),locals().values()):
		writer.add_text(f"locals/{k}", f"{v}")
		logger.info(f"locals/{k} --> {v}")
	print("===========================================")


	# ================== training loop ==================
	model.train()
	model = model.to(device)
	batch_iter = 0
	num_epochs_worse = 0
	checkpoint_path = os.path.join(PROJECT_ROOT,"saved_data/checkpoints",log_name) + ".pth"
	path_items = log_name.split("/")
	if  len(path_items) > 1:
		Path(os.path.join(PROJECT_ROOT,"saved_data/checkpoints",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
	best_val_acc = 0.0

	logger.info(f"****************************************** Training Started ******************************************")

	for e in range(epochs):
		model.train()
		model = model.to(device)
		if num_epochs_worse == ese:
			break

		if e == 0:
			# at epoch 0 evaluate on the validation set
			val_acc,val_loss = validate(model, val_loader, device, loss_fn)
			writer.add_scalar(f"val_metric/val_loss", val_loss, e)
			writer.add_scalar(f"val_metric/val_acc", val_acc, e)

			# logging
			logger.info('Train Epoch: {}, val_acc: {:.3f}, val loss: {:.3f}'.format(e, val_acc, val_loss))

		# for batch_idx, (data,target) in enumerate(train_loader):
		for batch_idx, (data,target) in enumerate(train_loader):
			# stop training, run on the test set
			if num_epochs_worse == ese:
				break

			# generic batch processing
			data,target = data.to(device),target.to(device)

			# forward pass
			output = model(data)

			# loss
			train_loss = loss_fn(output,target)
			writer.add_scalar(f"train_metric/loss", train_loss, batch_iter)

			# backward pass
			train_loss.backward()

			# step
			optimizer.step()
			optimizer.zero_grad()

			# logging
			if batch_idx % log_freq == 0:
				if (100.0 * (batch_idx+1) / len(train_loader)) == 100:
					logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
								e, len(train_loader.dataset), len(train_loader.dataset),
								100.0 * (batch_idx+1) / len(train_loader), train_loss))
				else:
					logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
								e, (batch_idx+1) * train_loader.batch_size, len(train_loader.dataset),
								100.0 * (batch_idx+1) / len(train_loader), train_loss))
			batch_iter += 1

		# at end of epoch evaluate on the validation set
		val_acc,val_loss = validate(model, val_loader, device, loss_fn)
		writer.add_scalar(f"val_metric/val_loss", val_loss, e)
		writer.add_scalar(f"val_metric/val_acc", val_acc, e)

		# logging
		logger.info('Train Epoch: {}, val_acc: {:.3f}, val loss: {:.3f}'.format(e, val_acc, val_loss))

		# check if to save new chckpoint
		if best_val_acc < val_acc:
			logger.info("==================== best validation metric ====================")
			logger.info('Train Epoch: {}, val_acc: {:.3f}, val loss: {:.3f}'.format(e,val_acc, val_loss))
			best_val_acc = val_acc

			torch.save({
				'epoch': e + 1,
				'model_state_dict': model.state_dict(),
				f"val_acc": val_acc,
				'val_loss': val_loss,
			}, checkpoint_path)
			num_epochs_worse = 0
		else:
			logger.info(f"info: {num_epochs_worse} num epochs without improving")
			num_epochs_worse += 1

		# check for early stopping
		if num_epochs_worse == ese:
			logger.info(f"Stopping training because validation metric did not improve after {num_epochs_worse} epochs")
			break

		if lr_scheduler is not None:
			lr_scheduler.step()

	logger.info(f"Best val acc: {best_val_acc}")
	model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
	

	logger.info("========================= Training Finished =========================")


def validate(model, val_loader, device, loss_fn):
	model.eval()
	model = model.to(device)

	val_loss = 0

	# collect all labels and predictions, then feed to val metric specific function
	with torch.no_grad():
		predictions = []
		labels = []
		outputs = []

		for batch_idx, (data,target) in enumerate(tqdm(val_loader)):
			# parse the batch and send to device
			data,target = data.to(device),target.to(device)

			# forward pass
			out = model(data)

			# get the loss
			val_loss += loss_fn(out, target)

			# parse the output for the prediction
			prediction = out.argmax(dim=1).to('cpu')

			predictions.append(prediction)
			labels.append(target.to('cpu'))
			outputs.append(out.to('cpu'))
		
		predictions = torch.cat(predictions).numpy()
		labels = torch.cat(labels).numpy()
		outputs = torch.cat(outputs)

		val_loss /= (len(val_loader))
		val_acc = (predictions == labels).mean()

		return val_acc, val_loss
	

if __name__ == '__main__':
	# ============ Argument parser ============
	parser = argparse.ArgumentParser(description='Finetune Model')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
	parser.add_argument('--seed', type=int, default=123, help='random seed')
	parser.add_argument('--logname', type=str, default='motion_blur', help='name of experiment')
	parser.add_argument('--dataset', type=str, default='cifar10', help='(cifar10)')
	parser.add_argument('--model_name', type=str, default='Standard', help='model architecture')
	parser.add_argument('--corr', type=str, default='gaussian_noise', help='corruption')
	args = parser.parse_args()

	# set up training
	init_seeds(args.seed)
	logging_prefix = args.corr
	log_name = args.corr+str(args.seed)
	logger = init_logger(f"{logging_prefix}/seed{args.seed}")
	logger.info(args)
	
	# init model, loss, optimizer, learning rate schedule
	model = load_model(model_name=args.model_name,dataset=args.dataset,threat_model='corruptions').eval()

	for name, param in model.named_parameters():
		# if name == 'fc.weight':
		# 	init.xavier_uniform_(param)
		# if name == 'conv1.weight':
		# 	init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
		
		if name == 'bn1.weight' or name == 'block3.layer.3.bn2.weight':
			init.ones_(param)
		elif name == 'bn.bias' or name == 'block3.layer.3.bn2.bias':
			init.zeros_(param)
		else:
			param.requires_grad = False
			

	criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4)
	# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
	
	# load the data
	train_ds, val_ds, test_ds = get_cifar10c_data([args.corr],1000)

	logger.info(f"val_length: {len(train_ds)}")
	logger.info(f"val_length: {len(val_ds)}")
	logger.info(f"test_length: {len(test_ds)}")

	dl = DataLoader(train_ds,args.batch_size,shuffle=True)
	dl_v = DataLoader(val_ds,batch_size=128)
	dl_t = DataLoader(test_ds,batch_size=256)

	train(model,criterion,optimizer,log_name,args.epochs,100,'cuda',dl,dl_v,logger,lr_sch,5)
	logger.info("========================= Test Results =========================")

	# checkpoint_path = os.path.join(PROJECT_ROOT,"saved_data/checkpoints",log_name) + ".pth"
	# model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
	# test_acc,test_loss = validate(model,dl_t,'cuda',criterion)
	# logger.info(f'Accuracy: {test_acc}, Loss: {test_loss}')