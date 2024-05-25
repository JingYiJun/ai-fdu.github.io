import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import h5py
import cv2
import shutil
from model import CSRNet


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar', save_dir='./model/'):  # 添加保存目录参数
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(save_dir, task_id + 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)


def load_data(img_path, gt_path, train=True):
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(
        target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64
    return img, target


class ImgDataset(Dataset):
    def __init__(self, img_dir, gt_dir, shape=None, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(
            img_dir) if filename.endswith('.jpg')]

        if shuffle:
            random.shuffle(self.img_paths)

        self.nSamples = len(self.img_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(
            self.gt_dir, os.path.splitext(img_name)[0] + '.h5')
        img, target = load_data(img_path, gt_path, self.train)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


lr = 1e-5
original_lr = lr
batch_size = 1
# momentum = 0.95
decay = 5*1e-4
num_epochs = 400
steps = [-1, 1, 100, 150]
scales = [1, 1, 1, 1]
workers = 4
seed = time.time()
print_freq = 30
img_dir = "./dataset/train/rgb/"
gt_dir = "./dataset/train/hdf5s/"
pre = "./model/model_best.pth.tar"
task = ""


def main():
    start_epoch = 0
    best_prec1 = 1e6

    torch.cuda.manual_seed(seed)

    model = CSRNet()

    model = model.cuda()

    criterion = nn.MSELoss(reduction="sum").cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False, min_lr=1e-7)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImgDataset(img_dir, gt_dir, transform=transform, train=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    if pre:
        if os.path.isfile(pre):
            print(f"=> loading checkpoint '{pre}'", flush=True)
            checkpoint = torch.load(pre)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{pre}' (start_epoch {start_epoch})", flush=True)
        else:
            print(f"=> no checkpoint found at '{pre}'", flush=True)

    for epoch in range(start_epoch, num_epochs):

        # adjust_learning_rate(optimizer, epoch)

        train(model, criterion, optimizer, epoch, train_loader)
        prec1 = validate(model, val_loader)
        scheduler.step(prec1)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(f' * best MAE {best_prec1:.3f} ', flush=True)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, task)


def train(model, criterion, optimizer, epoch, train_loader):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    current_lr = optimizer.param_groups[0]['lr']

    print(f'epoch {epoch}, processed {epoch * len(train_loader.dataset)} samples, lr {current_lr}', flush=True)

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t',
                  flush=True)


def validate(model, val_loader):
    print('begin test', flush=True)

    model.eval()
    mae = 0

    for i, (img, target) in enumerate(val_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() -
                   target.sum().type(torch.FloatTensor).cuda())

    mae = mae/len(val_loader)
    print(f' * MAE {mae:.3f} ', flush=True)

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = original_lr

    for i in range(len(steps)):

        scale = scales[i] if i < len(scales) else 1

        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()

# srun -p x090 --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:1 --job-name=python --nodes=1 --ntasks=1 --time=2:00:00 --output=train.log python train.py