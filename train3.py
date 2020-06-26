import argparse
import datetime
import glob
import os
import random
import shutil
import time
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm

from convnet3 import Convnet
from dataset2 import CellsDataset

parser = argparse.ArgumentParser('Predicting hits from pixels')
parser.add_argument('name',type=str,help='Name of experiment')
parser.add_argument('data_dir',type=str,help='Path to data directory containing images and gt.csv')
parser.add_argument('--num_steps',type=int,default=20000,help='Number of training iterations')
parser.add_argument('--batchsize',type=int,default=16,help='Size of batch')
parser.add_argument('--weight_decay',type=float,default=0.0,help='Weight decay coefficient (something like 10^-5)')
parser.add_argument('--lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--resume',action='store_true',help='Resume experiments from checkpoint directory')
parser.add_argument('--seed',type=int,default=1337,help='RNG seed')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# create output directory tree:
if os.path.isdir(args.name) and not args.resume:
    shutil.rmtree(args.name)
logs_path = join(args.name,'logs')
checkpoints_path = join(args.name,'checkpoints')
checkpoint_path = join(checkpoints_path,'checkpoint.pth')
if not os.path.isdir(args.name):
    os.mkdir(args.name)
    os.mkdir(logs_path)
    os.mkdir(checkpoints_path)

# record arguments for future reference:
with open(join(args.name,'arguments.txt'),'w') as fout:
    fout.write(str(args))

# load metadata:
metadata = pd.read_csv(join(args.data_dir,'gt.csv'))
metadata.set_index('filename', inplace=True)

# create datasets:
toTensor = ToTensor()
dataset_train = CellsDataset(args.data_dir,transform=ToTensor(),return_filenames=True)

random.shuffle(dataset_train.files)
split_point = int(len(dataset_train) * 0.9) # 90/10 train/val split
dataset_test = CellsDataset(args.data_dir,transform=ToTensor(),return_filenames=True)

dataset_test.files = dataset_train.files[split_point:]
dataset_train.files = dataset_train.files[:split_point]

loader_train = DataLoader(dataset_train,batch_size=args.batchsize,shuffle=True,num_workers=4,pin_memory=True)
loader_test =  DataLoader(dataset_test, batch_size=args.batchsize,shuffle=True,num_workers=4,pin_memory=True)


# create model:
model = Convnet()
model.to(device)

# create optimizer:
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

if args.resume:
    try:
        checkpoint = torch.load(checkpoint_path)
        print('Resuming from checkpoint...')
        model.load_state_dict(checkpoint['state_dict'])
        globalStep = checkpoint['globalStep']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dataset_train.files = checkpoint['train_paths']
        dataset_test.files = checkpoint['test_paths']
    except FileNotFoundError:
        globalStep = 0

# create logger:
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
writer = SummaryWriter()

# main training loop
global_step = 0
best_test_error = 10000
for epoch in range(15):
    print("Epoch %d" % epoch)
    model.train()
    for images, paths in tqdm(loader_train):
        images = images.to(device)
        targets = torch.tensor([metadata['count'][os.path.split(path)[-1]] for path in paths]) # B
        targets = targets.float().to(device)

        # forward pass:
        output = model(images) # B x 1 x 9 x 9 (analogous to a heatmap)
        preds = output.sum(dim=[1,2,3]) # predicted cell counts (vector of length B)
        
        # backward pass:
        loss = torch.mean((preds - targets)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging:
        count_error = torch.abs(preds - targets).mean()
        writer.add_scalar('train_loss', loss.item(), global_step=global_step)
        writer.add_scalar('train_count_error', count_error.item(), global_step=global_step)

        print("Step %d, loss=%f, count error=%f" % (global_step,loss.item(),count_error.item()))

        global_step += 1
    
    mean_test_error = 0
    model.eval()
    for images, paths in tqdm(loader_test):
        images = images.to(device)
        targets = torch.tensor([metadata['count'][os.path.split(path)[-1]] for path in paths]) # B
        targets = targets.float().to(device)

        # forward pass:
        output = model(images) # B x 1 x 9 x 9 (analogous to a heatmap)
        preds = output.sum(dim=[1,2,3]) # predicted cell counts (vector of length B)

        # logging:
        loss = torch.mean((preds - targets)**2)
        count_error = torch.abs(preds - targets).mean()
        mean_test_error += count_error
        writer.add_scalar('test_loss', loss.item(), global_step=global_step)
        writer.add_scalar('test_count_error', count_error.item(), global_step=global_step)
        
        global_step += 1
    
    mean_test_error = mean_test_error / len(loader_test)
    print("Test count error: %f" % mean_test_error)
    if mean_test_error < best_test_error:
        best_test_error = mean_test_error
        torch.save({'state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'globalStep':global_step,
                    'train_paths':dataset_train.files,
                    'test_paths':dataset_test.files},checkpoint_path)