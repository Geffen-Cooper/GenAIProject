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
import datetime
import json
from pathlib import Path
from collections import OrderedDict

from datasets import *
import sys
sys.path.append("../")
from utils.setup_funcs import init_logger, init_seeds
from utils.setup_funcs import PROJECT_ROOT

from robustbench.utils import clean_accuracy
from robustbench.utils import load_model

np.set_printoptions(linewidth=np.nan)


def train(model,loss_fn,optimizer,log_name,epochs,ese,device,
		  train_loader,val_loader,logger,lr_scheduler,log_freq,metrics):

	# start tensorboard session
	path_items = log_name.split("/")
	if  len(path_items) > 1:
		Path(os.path.join(PROJECT_ROOT,"saved_data/runs",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(os.path.join(PROJECT_ROOT,"saved_data/runs",*path_items))

	# log training parameters
	print("===========================================")
	for k,v in zip(locals().keys(),locals().values()):
		# writer.add_text(f"locals/{k}", f"{v}")
		logger.info(f"locals/{k} --> {v}")
	print("===========================================")


	# ================== training loop ==================
	model.eval()
	model = model.to(device)
	batch_iter = 0
	num_epochs_worse = 0
	
	Path(os.path.join(PROJECT_ROOT,"saved_data/checkpoints",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
	checkpoint_path = os.path.join(PROJECT_ROOT,"saved_data/checkpoints",*path_items[:-1],path_items[-1]+".pth")
	best_val_acc = 0.0

	logger.info(f"****************************************** Training Started ******************************************")
	
	for e in range(epochs):
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
		writer.add_scalar(f"val_metric/val_loss", val_loss, e+1)
		writer.add_scalar(f"val_metric/val_acc", val_acc, e+1)

		# logging
		logger.info('Train Epoch: {}, val_acc: {:.3f}, val loss: {:.3f}'.format(e, val_acc, val_loss))

		# check if to save new chckpoint
		if best_val_acc < val_acc:
			logger.info("==================== best validation metric ====================")
			logger.info('Train Epoch: {}, val_acc: {:.3f}, val loss: {:.3f}'.format(e,val_acc, val_loss))
			best_val_acc = val_acc

			if metrics['subset_state_dict_keys'] != 'all':
				torch.save({
					'epoch': e + 1,
					'model_state_dict': OrderedDict((key, value.cpu()) for key, value in model.state_dict().items() if key in metrics['subset_state_dict_keys']),
					'val_acc': val_acc,
					'val_loss': val_loss.cpu(),
				}, checkpoint_path)
			else:
					torch.save({
					'epoch': e + 1,
					'model_state_dict': model.state_dict(),
					'val_acc': val_acc,
					'val_loss': val_loss.cpu(),
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
	metrics['best_val_acc'] = best_val_acc
	model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'],strict=False)
	

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
	script_start = time.time()
	# ============ Argument parser ============
	parser = argparse.ArgumentParser(description='Finetune Model')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
	parser.add_argument('--seed', type=int, default=123, help='random seed')
	parser.add_argument('--dataset', type=str, default='cifar10', help='(cifar10)')
	parser.add_argument('--model_name', type=str, default='Standard', help='model architecture')
	parser.add_argument('--corr', type=str, default='frost', help='corruption')
	parser.add_argument('--finetune_config',type=str,default='fc',help='which modules to finetune')
	parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file (default: None)")
	parser.add_argument("--logname", type=str, help="optional logname")
	args = parser.parse_args()

	# set up training
	init_seeds(args.seed)
	if args.logname is None:
		args.logname = args.finetune_config
	log_name = os.path.join(args.logname,args.corr,"seed"+str(args.seed)+"_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
	logger = init_logger(log_name)
	logger.info(args)

	# save metrics
	path_items = log_name.split("/")
	if  len(path_items) > 1:
		Path(os.path.join(PROJECT_ROOT,"saved_data/metrics",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
	metrics = {'subset_state_dict_keys':None,'test_acc':0.,'train_time':0.,'num_params':0}
	
	# init model, loss, optimizer, learning rate schedule
	model = load_model(model_name=args.model_name,dataset=args.dataset,threat_model='corruptions').eval()

	if not args.finetune_config == 'all':
		for name, module in model.named_modules():
			if args.finetune_config == 'first_conv':
				metrics['subset_state_dict_keys'] = ['conv1.weight']
				if name == 'conv1':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				else: # freezeall other parameters
					for param in module.parameters():
						param.requires_grad = False
			
			elif args.finetune_config == 'fc':
				metrics['subset_state_dict_keys'] = ['linear.weight','linear.bias']
				if name == 'fc':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				else: # freezeall other parameters
					for param in module.parameters():
						param.requires_grad = False
					
			elif args.finetune_config == "last_three_bn":
				metrics['subset_state_dict_keys'] = ['bn1.weight','bn1.bias','block3.layer.3.bn1.weight','block3.layer.3.bn1.bias','block3.layer.3.bn2.weight','block3.layer.3.bn2.bias']
				if name == 'bn1':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				elif name == 'block3.layer.3.bn1':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				elif name == 'block3.layer.3.bn2':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				else: # freezeall other parameters
					for param in module.parameters():
						param.requires_grad = False	

			elif args.finetune_config == "last_two_bn":
				metrics['subset_state_dict_keys'] = ['bn1.weight','bn1.bias','block3.layer.3.bn2.weight','block3.layer.3.bn2.bias']
				if name == 'bn1':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				elif name == 'block3.layer.3.bn2':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					module.reset_parameters()
					for param in module.parameters():
						param.requires_grad = True
				else: # freezeall other parameters
					for param in module.parameters():
						param.requires_grad = False

			elif args.finetune_config == "last_bn":
				metrics['subset_state_dict_keys'] = ['bn1.weight','bn1.bias']
				if name == 'bn1':
					metrics['num_params'] += sum(p.numel() for p in module.parameters())
					# module.reset_parameters()
					nn.init.normal_(module.weight.data, mean=0, std=0.1)  # Initialize weights from a normal distribution
					nn.init.normal_(module.bias.data, mean=0, std=0.1)
					for param in module.parameters():
						param.requires_grad = True
				else: # freezeall other parameters
					for param in module.parameters():
						param.requires_grad = False	
	else:
		metrics['subset_state_dict_keys'] = 'all'


	criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4)
	# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
	
	# load the data
	# train_ds, val_ds, test_ds = get_cifar10c_data([args.corr],5000)
	train_ds, val_ds, test_ds = get_cifar10c_data(["gaussian_noise","frost","contrast","pixelate","glass_blur"],1000)
	# print(len(train_ds),len(val_ds),len(test_ds))
	# exit()

	logger.info(f"train_length: {len(train_ds)}")
	logger.info(f"val_length: {len(val_ds)}")
	logger.info(f"test_length: {len(test_ds)}")

	dl = DataLoader(train_ds,args.batch_size,shuffle=True)
	dl_v = DataLoader(val_ds,batch_size=128)
	dl_t = DataLoader(test_ds,batch_size=256)

	if not args.finetune_config == 'none':	
		train(model,criterion,optimizer,log_name,args.epochs,100,'cuda',dl,dl_v,logger,lr_sch,5,metrics)
		checkpoint_path = os.path.join(PROJECT_ROOT,"saved_data/checkpoints",log_name) + ".pth"
		model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'],strict=False)
	logger.info("========================= Test Results =========================")

	test_acc,test_loss = validate(model,dl_t,'cuda',criterion)
	logger.info(f'Accuracy: {test_acc}, Loss: {test_loss}')

	script_end = time.time()
	metrics['test_acc'] = test_acc
	metrics['train_time'] = script_end-script_start
	with open(os.path.join(PROJECT_ROOT,"saved_data/metrics",*path_items)+'.json', 'w') as f:
		json.dump(metrics, f, indent=4)
	