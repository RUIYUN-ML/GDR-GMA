import argparse
import os
import time

import copy
import torch
import torch.nn as nn
from datas.dataLoaders import DataLoaders
from models.resnet import resnet18
from models.tinyvit import tiny_vit_5m_224
from seed import set_seed
from torch.optim.lr_scheduler import MultiStepLR
from utils import *
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# parameters setting
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--name', default='train_model', help='filename to output best model') #save output
parser.add_argument('--model', default='resnet-18',help="models e.g. resnet18|vit")
parser.add_argument('--dataset', default='cifar-10',help="datasets e.g. cifar-10|imagenet")
parser.add_argument('--batch_size', default=256,type=int, help='batch size')
parser.add_argument('--epoch', default=200,type=int, help='epoch')
parser.add_argument('--exp_dir',default='')
parser.add_argument('--lr', default=0.1, type=float, help="learning rate for normal path")
parser.add_argument('--train', default='True', type=str, help='train or test the model')
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


def val_model(model:nn.Module, criterion, test_loader):
    model.eval()
    val_loss = 0.0
    val_corrects = 0
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
            
            
def train_model(model:nn.Module, criterion, optimizer, scheduler, data_loaders, num_epochs=200):
    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        begin_time = time.time()
        running_loss = 0.0
        running_corrects = 0.0
        dataset_sizes = len(data_loaders['train'].dataset)
        for idx, (data, targets) in enumerate(data_loaders['train']):
            data, targets = data.to(args.device), targets.to(args.device)
            outputs = model(data)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            
            preds = torch.argmax(outputs.data, dim=1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_corrects += torch.sum(preds == targets.data)
            
        epoch_loss = running_loss / dataset_sizes
        epoch_acc = float(running_corrects) / dataset_sizes
        
        scheduler.step()
        
        val_loss, val_acc = val_model(model, criterion, data_loaders['val'])
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        print(f'Epoch: {epoch} - train_loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}- time: {time.time() - begin_time:.2f}s | best_epoch: {best_epoch} - best_acc: {best_acc:.4f}')

    
        checkpoint_dir = os.path.join(args.exp_dir, args.dataset + '-' + args.model)
        if args.load_checkpoint == '':
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch_%d.pt' % epoch)
        
 
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_acc': best_acc,
                'best_train_acc': best_train_acc,
                'best_epoch': best_epoch,
            }, checkpoint_path)
    
        update_checkpoint_link(checkpoint_dir, [('epoch_%d.pt' % best_epoch, 'best.pt'),('epoch_%d.pt' % epoch, 'last.pt')], num_epochs)
        
    
    
    
    
    
if __name__=="__main__":
    if args.model == 'resnet18':
        model = resnet18()
    elif args.model == 'vit':
        model = tiny_vit_5m_224()


    dataloaders = DataLoaders(args.dataset, batch_size=args.batch_size, seed=args.seed).load_data()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

    model = torch.nn.DataParallel(model) # device_ids=args.gpu_ids
    model = model.to(args.device) # args.gpu_ids[0]
        
    train_model(model, criterion, optimizer, scheduler, dataloaders, args.epoch)
        
        
    