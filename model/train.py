import os

from torch._C import device
from model import PHNet
from utils import save_checkpoint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import json
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch PHNet')

# two dashes -- means not necessary argument
parser.add_argument('--train_json', metavar='TRAIN', help='path to train json', default="./jsons/train10.json")
parser.add_argument('--test_json', metavar='TEST', help='path to test json', default="./jsons/test10.json")
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' , type=int, help='batch size', default=30)
parser.add_argument('--gpu',metavar='GPU', type=str, help='GPU id to use.', default="5")
parser.add_argument('--task',metavar='TASK', type=str, help='task id to use.', default="1")
parser.add_argument('--gt_code', metavar='GT_NUMBER' ,type=str, help='ground truth dataset number', default='4896')
parser.add_argument('--logname',metavar='LOGNAME', type=str, help='Log name to save.')

def GAME(img, target, level = 1):
    batch_size = img.shape[0]
    w, h = img.shape[2], img.shape[3]
    w, h = w//(level + 1), h//(level + 1)
    game = 0
    for batch in range(batch_size):
        game += abs(img[batch,:,:,:].sum() - target[batch,:,:,:].sum())
    return game

def main():
    global args, best_prec1, device
    device='cpu'
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 1 # changed by me
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]

    # multiprocessing
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    args.logname = args.task

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)

    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    model = PHNet()
    model = model.to(device)
    
    # criterion = nn.MSELoss(size_average=False).to(device)
    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)
    model = DataParallel_withLoss(model, criterion)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_dict = model.state_dict()
            pretrained_dict = {k : v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)
        with open("./"+args.logname+".txt", "a") as f:
            f.write("epoch " + str(epoch) + "  mae: " +str(float(prec1)))
            f.write("\n")
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task, epoch = epoch,path='./ckpt/'+args.task)

def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(
                        root = train_list, 
                        shuffle = True, 
                        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ] ),
                        train = True,
                        gt_code = args.gt_code,
                        batch_size = args.batch_size,
                        num_workers = args.workers),
                        batch_size = args.batch_size
        )

    print(f"epoch {epoch}, processed {epoch * len(train_loader.dataset)} samples, lr {args.lr}")
    model.train()
    end = time.time()
    for i,(img, target) in enumerate(train_loader):
        # img.shape (batch, rgb, frames, height, width)
        # target.shape (batch, shrinked_height, shrinked_width), the resized gaussian filter result
        # check image.py for how it shrinks

        data_time.update(time.time() - end)

        # cast image to float type, and move to device
        img = img.to(device)
        img = img.type(torch.FloatTensor)
        # cast target to float type, and move to device
        target = target.type(torch.FloatTensor).unsqueeze(1).to(device)

        loss,_ = model(target, img)
        loss = loss.sum()
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   gt_code = args.gt_code,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
                     ,train=False),
    batch_size=args.batch_size)
    model.eval()
    mae = 0
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            img = img.to(device)
            img = img.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).unsqueeze(1).to(device)
            _, output = model(target, img)
            mae += GAME(output.data, target, 0)
    mae = mae/len(test_loader)/args.batch_size
    print(' * MAE {mae:.3f} '.format(mae=mae))
    return mae

def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(inputs[0])
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss,0),outputs

def DataParallel_withLoss(model, loss, **kwargs):
    # multi processing stuff
    model = FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids = kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device = kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID = kwargs['cuda']
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    return model

if __name__ == '__main__':
    main()
