import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models.resnet import resnet32x4, resnet8x4

import gc
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore")
# from imagenet_train_cfg import cfg as config
# from tools import utils

from dataset.cifar100 import get_cifar100_dataloaders

PATH_TEACHER = '/kaggle/input/trained-resnet/teacher_resnet32x4_acc79.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_lr = 0.1
weight_decay = 1e-4
momentum = 0.9
epochs = 200

class WSLDistiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(WSLDistiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

        self.T = 4
        self.alpha = 2.25

        self.hard_loss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.softmax = nn.Softmax(dim=1).cuda()
            self.logsoftmax = nn.LogSoftmax().cuda()
            self.hard_loss = self.hard_loss.cuda()


    def forward(self, x, label):

        fc_t = self.t_net(x)
        fc_s = self.s_net(x)

        s_input_for_softmax = fc_s / self.T
        t_input_for_softmax = fc_t / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = - torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)

        fc_s_auto = fc_s.detach()
        fc_t_auto = fc_t.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(label, num_classes=1000).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

        hard_loss = self.hard_loss(fc_s, label)

        loss = hard_loss + self.alpha * soft_loss

        return fc_s, loss

class data_prefetcher():
    def __init__(self, loader, mean=None, std=None, is_cutout=False, cutout_length=16, is_sample=False):
        self.is_sample = is_sample
        self.loader = iter(loader)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        if mean is None:
            self.mean = torch.tensor([0.5071 * 255, 0.4867 * 255, 0.4408 * 255]).cuda().view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([m * 255 for m in mean]).cuda().view(1, 3, 1, 1)
        if std is None:
            self.std = torch.tensor([0.2675 * 255, 0.2565 * 255, 0.2761 * 255]).cuda().view(1, 3, 1, 1)
        else:
            self.std = torch.tensor([s * 255 for s in std]).cuda().view(1, 3, 1, 1)
        self.is_cutout = is_cutout
        self.cutout_length = cutout_length
        self.preload()

    def normalize(self, data):
        data = data.float()
        data = data.sub_(self.mean).div_(self.std)
        return data

    def cutout(self, data):
        batch_size, h, w = data.shape[0], data.shape[2], data.shape[3]
        mask = torch.ones(batch_size, h, w).cuda()
        y = torch.randint(low=0, high=h, size=(batch_size,))
        x = torch.randint(low=0, high=w, size=(batch_size,))

        y1 = torch.clamp(y - self.cutout_length // 2, 0, h)
        y2 = torch.clamp(y + self.cutout_length // 2, 0, h)
        x1 = torch.clamp(x - self.cutout_length // 2, 0, w)
        x2 = torch.clamp(x + self.cutout_length // 2, 0, w)
        for i in range(batch_size):
            mask[i][y1[i]: y2[i], x1[i]: x2[i]] = 0.
        mask = mask.expand_as(data.transpose(0, 1)).transpose(0, 1)
        data *= mask
        return data

    def preload(self):
        try:
            if self.is_sample:
                self.next_input, self.next_target, self.next_index, self.next_sample_index = next(self.loader)
            else:
                self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.normalize(self.next_input)
            if self.is_cutout:
                self.next_input = self.cutout(self.next_input)

            if self.is_sample:
                self.next_index = self.next_index.cuda(non_blocking=True)
                self.next_sample_index = self.next_sample_index.cuda(non_blocking=True)

    def next(self):

        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())

        self.preload()

        if self.is_sample:
            index = self.next_index
            sample_index = self.next_sample_index
            if index is not None:
                index.record_stream(torch.cuda.current_stream())
            if sample_index is not None:
                sample_index.record_stream(torch.cuda.current_stream())
            return input, target, index, sample_index
        else:
            return input, target


def build_dummy_dataset():
    """Build a dummy dataset for testing purposes."""
    # create some random data have shape (16, 3, 32, 32)
    # create dataloader for train, valid, test
    num_classes = 5

    train_data = torch.randn(16, 3, 32, 32)
    train_labels = torch.randint(0, num_classes, (16,))

    train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=4)
    valid_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=4)
    test_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=4)

    return train_loader, valid_loader, test_loader

best_err1 = 100
best_err5 = 100

def main():
    global best_err1, best_err5

    train_loader, val_loader = get_cifar100_dataloaders(batch_size=128, num_workers=4)
    # train_loader, val_loader, test_loader = build_dummy_dataset()

    t_net = resnet32x4(num_classes=100)
    try:
        t_net.load_state_dict(torch.load(PATH_TEACHER, map_location=device)['net'])
    except Exception as e:
        print("Error loading teacher model:", e)
    s_net = resnet8x4(num_classes=100)

    d_net = WSLDistiller(t_net, s_net)

    t_net = torch.nn.DataParallel(t_net)
    s_net = torch.nn.DataParallel(s_net)
    d_net = torch.nn.DataParallel(d_net)

    t_net = t_net.to(device)
    s_net = s_net.to(device)
    d_net = d_net.to(device)

    ### choose optimizer parameters

    optimizer = torch.optim.SGD(s_net.parameters(), lr=init_lr,  # config.optim.init_lr
                                momentum=momentum, weight_decay=weight_decay, nesterov=True)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.enabled = True

    validate(val_loader, t_net, 0)

    for epoch in range(epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_with_distill(train_loader, d_net, optimizer, epoch)

        # evaluate on validation set
        err1, err5 = validate(val_loader, s_net, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
            print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
            ckt = {
                'epoch': epoch,
                'state_dict': s_net.module.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
            }
            save_path = 'best8x4_wsl.pth'
            torch.save(ckt, save_path)
        gc.collect()

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    step = 0
    while input is not None:
        # for PyTorch 0.4.x, volatile=True is replaced by with torch.no.grad(), so uncomment the followings:
        with torch.no_grad():
            output = model(input)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 100 == 0:
            print('Test (on val set): [Epoch {0}/{1}][Batch {2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, epochs, step, len(val_loader), batch_time=batch_time, top1=top1, top5=top5))
        input, target = prefetcher.next()
        step += 1

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'
          .format(epoch, epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg


def train_with_distill(train_loader, d_net, optimizer, epoch):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    btic = time.time()

    prefetcher = data_prefetcher(train_loader, is_sample=False)
    inputs, targets = prefetcher.next()

    step = 0
    while inputs is not None:

        batch_size = inputs.size(0)
        if step == 0:
            print('epoch %d lr %e' % (epoch, optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()

        outputs, loss = d_net(inputs, targets)

        loss = torch.mean(loss)
        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            speed = 100 * batch_size / (time.time() - btic)
            print(
                'Train with distillation: [Epoch %d/%d][Batch %d/%d]\t, speed %.3f, Loss %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                (epoch, epochs, step, len(train_loader), speed, train_loss.avg, top1.avg, top5.avg))
            btic = time.time()

        inputs, targets = prefetcher.next()

        step += 1


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)


        # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        if k == 1:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        else:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            
        # correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()