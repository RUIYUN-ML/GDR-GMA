import argparse
import copy
import os
import time


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datas.dataLoaders import DataLoaders
from models.resnet import resnet18
from seed import set_seed
from metrics.MIA_LR import get_membership_attack_prob
from models.tinyvit import tiny_vit_5m_224
import random

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# parameters setting
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--name', default='train_model', help='filename to output best model')
parser.add_argument('--model', default='resnet18',help="models e.g. resnet18|vit")
parser.add_argument('--dataset', default='cifar-10',help="datasets e.g. cifar-10|imagenet")
parser.add_argument('--batch_size', default=256,type=int, help='batch size')
parser.add_argument('--unlearned_size', default=0.2,type=float, help='unlearned size')
parser.add_argument('--seed', default=0, type=int, help='random seed for the entire program')
parser.add_argument('--cudnn_behavoir', default='benchmark', type=str, help='cudnn behavoir [benchmark|normal(default)|slow|none] from left to right, cudnn randomness decreases, speed decreases')
parser.add_argument('--load_checkpoint', default='', type=str, help='path to load a checkpoint')


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1
    
set_seed(args.seed, args.cudnn_behavoir)

print('args')
for arg in vars(args):
     print('   ',arg, '=' ,getattr(args, arg))
print()

def val_model(model:nn.Module, test_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    val_loss = 0.0
    val_corrects = 0.0
    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(args.device), targets.to(args.device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            pred = torch.argmax(outputs, dim=1)
            
            val_loss += loss.item()
            val_corrects += torch.sum(pred == targets)

    val_loss /= len(test_loader.dataset)
    val_acc = val_corrects.double() / len(test_loader.dataset)
    
    return val_loss, val_acc


if __name__=="__main__":
    
    if args.model == 'resnet18':
        model = resnet18()
    elif args.model == 'vit':
        model = tiny_vit_5m_224()


    dataloaders = DataLoaders(args.dataset, batch_size=args.batch_size, unlearned_size=args.unlearned_size, seed=args.seed).load_unlearn_data()
    model = torch.nn.DataParallel(model) # device_ids=args.gpu_ids
    model = model.to(args.device) # args.gpu_ids[0]
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])



    n_loss, n_acc = val_model(model, dataloaders['remain'])
    t_loss, t_acc = val_model(model, dataloaders['unlearn'])
    val_loss, val_acc = val_model(model, dataloaders['val'])
    mia = get_membership_attack_prob(dataloaders['remain'], dataloaders['unlearn'], dataloaders['val'], model)

    print(f't_loss: {t_loss:.4f}, t_acc: {t_acc:.4f}, n_loss: {n_loss:.4f}, n_acc: {n_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, MIA :{mia:.4f}')

