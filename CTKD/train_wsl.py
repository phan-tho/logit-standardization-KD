import json
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models.resnet import resnet32x4, resnet8x4

import gc
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
name_file_log = 'log_wsl.json'

class WSLDistiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(WSLDistiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

        self.T = 4
        self.alpha = 2.25

        self.last_epoch = 0

        self.hard_loss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.softmax = nn.Softmax(dim=1).cuda()
            self.logsoftmax = nn.LogSoftmax().cuda()
            self.hard_loss = self.hard_loss.cuda()


    def forward(self, x, label, epoch):
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

        one_hot_label = F.one_hot(label, num_classes=100).float()
        # one_hot_label shape (batch_size, num_classes)
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)
        # softmax_loss_s and softmax_loss_t shape (batch_size, 1)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

        hard_loss = self.hard_loss(fc_s, label)

        loss = hard_loss + self.alpha * soft_loss

        if epoch != self.last_epoch:
            self.last_epoch = epoch
            if epoch % 10 == 0:
                focal_weight_log = focal_weight.detach().cpu().numpy().tolist()
                softmax_loss_s_log = softmax_loss_s.detach().cpu().numpy().tolist()
                softmax_loss_t_log = softmax_loss_t.detach().cpu().numpy().tolist()
                soft_loss_log = soft_loss.detach().cpu().numpy().tolist()
                hard_loss_log = hard_loss.detach().cpu().numpy().tolist()
                loss_log = loss.detach().cpu().numpy().tolist()
                log = {
                    'epoch': epoch,
                    'focal_weight': focal_weight_log,
                    'softmax_loss_s': softmax_loss_s_log,
                    'softmax_loss_t': softmax_loss_t_log,
                    'soft_loss': soft_loss_log,
                    'hard_loss': hard_loss_log,
                    'loss': loss_log
                }
                # save log to json file
                with open(name_file_log, 'r+') as f:
                    data = json.load(f)
                    data.update(log)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()

                gc.collect()
        return fc_s, loss

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

def load_model():
    t_net = resnet32x4(num_classes=100)
    try:
        t_net.load_state_dict(torch.load(PATH_TEACHER, map_location=device)['net'])
    except Exception as e:
        print("Error loading teacher model:", e)
    s_net = resnet8x4(num_classes=100)

    d_net = WSLDistiller(t_net, s_net)

    # t_net = torch.nn.DataParallel(t_net)
    # s_net = torch.nn.DataParallel(s_net)
    # d_net = torch.nn.DataParallel(d_net)

    t_net = t_net.to(device)
    s_net = s_net.to(device)
    d_net = d_net.to(device)

    return t_net, s_net, d_net


def main():
    best_acc = 0.0
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=128, num_workers=4)
    # train_loader, val_loader, test_loader = build_dummy_dataset()

    t_net, s_net, d_net = load_model()

    optimizer = torch.optim.SGD(s_net.parameters(), lr=init_lr,  # config.optim.init_lr
                                momentum=momentum, weight_decay=weight_decay, nesterov=True)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.enabled = True

    for epoch in range(epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, d_net, optimizer, epoch)
        acc1 = test(s_net, val_loader, epoch)

        # remember best prec@1 and save checkpoint
        if acc1 > best_acc:
            best_acc = acc1
            print('Current best accuracy (top-1 and 5 error):', best_acc)
            ckt = {
                'epoch': epoch,
                'state_dict': s_net.state_dict(),
                'acc@1': best_acc
            }
            save_path = 'best8x4_wsl.pth'
            torch.save(ckt, save_path)
        gc.collect()

    print('Best accuracy (top-1 and 5 error):', best_acc)

def train(train_loader, d_net, optimizer, epoch):
    print('epoch %d lr %e' % (epoch, optimizer.param_groups[0]['lr']))

    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()

    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)
        optimizer.zero_grad()

        outputs, loss = d_net(inputs, targets, epoch)

        if isinstance(loss, float):
            loss = torch.tensor(loss, device=device)
        loss = torch.mean(loss)
        # TypeError: mean(): argument 'input' (position 1) must be Tensor, not float

        acc1, acc5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(
                'Train: [Epoch %d/%d][Batch %d/%d]\t, Loss %.3f, acc@1 %.3f, acc@5 %.3f' %
                (epoch, epochs, idx, len(train_loader), train_loss.avg, top1.avg, top5.avg))


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
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(model, test_loader, epoch):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    
    return acc

if __name__ == '__main__':
    # Set up logging. create a new log file if it doesn't exist
    if not os.path.exists(name_file_log):
        with open(name_file_log, 'w') as f:
            json.dump({}, f, indent=4)

    main()