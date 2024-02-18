# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import logging  # 引入logging模块
import os.path
import time
import os
import math
import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader
from torch.nn.parallel import DataParallel
from src.models.networks.resnet import generate_model
import time
import json
import torch.nn.functional as F
from utils import AverageMeter2 as AverageMeter
from utils import calculate_acc_sigmoid
from sklearn.metrics import precision_score, recall_score, f1_score


def load_model(model, checkpoint_path, multi_gpu=False):
    """
    通用加载模型函数。

    :param model: 要加载状态字典的PyTorch模型。
    :param checkpoint_path: 模型权重文件的路径。
    :param multi_gpu: 布尔值，指示是否使用多GPU加载模型。
    :return: 加载了权重的模型。
    """
    # 加载状态字典
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        state_dict = pretrain['state_dict']
    # 检查是否为多卡模型保存的状态字典
    if list(state_dict.keys())[0].startswith('module.'):
        # 移除'module.'前缀（多卡到单卡）
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    # 加载状态字典
    model.load_state_dict(state_dict)
    # 如果需要在多GPU上运行模型
    if multi_gpu:
        # 使用DataParallel封装模型
        model = nn.DataParallel(model)

    return model
def precision(pred, label):
    # 计算精确度
    pred = (pred > 0.5).float()
    return precision_score(label.cpu(), pred.cpu())

def recall(pred, label):
    # 计算召回率
    pred = (pred > 0.5).float()
    return recall_score(label.cpu(), pred.cpu())

def calculate_f1_score(pred, label):
    pred = (pred > 0.5).float()
    return f1_score(label.cpu(), pred.cpu())


class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, scheduler, args, summaryWriter):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs
        self.epoch = 0
        self.best_acc = 0
        self.args = args
        self.loss_function = torch.nn.BCELoss()
        self.summaryWriter = summaryWriter
        self.use_clip = False
        self.self_model()

    def __call__(self):
        if self.args.phase == 'train':
            for epoch in tqdm(range(self.epochs)):
                start = time.time()
                self.epoch = epoch+1
                self.train_one_epoch()
                self.num_params = sum([param.nelement() for param in self.model.parameters()])
                # self.scheduler.step()
                end = time.time()
                print("Epoch: {}, train time: {}".format(epoch, end - start))
                if epoch % 1 == 0:
                    self.evaluate()
        else:
            self.evaluate()

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        self.model.to(self.device)

    def calculate_metrics(self, pred, label, seg=None, mask=None):
        with torch.no_grad():
            dice = 0
            if seg is not None:
                seg = torch.sigmoid(seg)  # 将模型输出转换为概率值
                seg = (seg > 0.5).float()  # 应用阈值0.5进行二值化
                self.dice_metric(seg, mask)
                dice = self.dice_metric.aggregate().item()
                self.dice_metric.reset()
            precision_val = precision(pred, label)
            recall_val = recall(pred, label)
            f1_score_val = calculate_f1_score(pred, label)
            acc = calculate_acc_sigmoid(pred, label)
        return acc, precision_val, recall_val, f1_score_val, dice

    def update_meters(self, meters, values):
        for meter, value in zip(meters, values):
            meter.update(value)

    def reset_meters(self, meters):
        for meter in meters:
            meter.reset()
    def print_metrics(self, meters, prefix=""):
        metrics_str = ' '.join([f'{k}: {v.avg:.4f}' for k, v in meters.items()])
        print(f'{prefix} {metrics_str}')

    def log_metrics_to_tensorboard(self, metrics, epoch, stage_prefix=''):
        """
        将指标和损失值写入TensorBoard，区分损失和指标，以及训练和验证阶段。
        参数:
        - metrics (dict): 包含指标名称和值的字典。
        - epoch (int): 当前的epoch。
        - stage_prefix (str): 用于区分训练和验证阶段的前缀（如'Train'/'Val'）。
        - category_prefix (str): 用于区分损失和性能指标的前缀（如'Loss'/'Metric'）。
        """
        for name, meter in metrics.items():
            if 'loss' not in name.lower():
                category_prefix = 'Metric'
            else:
                category_prefix = 'Loss'
            tag = f'{category_prefix}/{name}'
            if 'lr' in name.lower():
                tag = 'lr'
            value = meter.avg if isinstance(meter, AverageMeter) else meter
            self.summaryWriter.add_scalars(tag, {stage_prefix: value}, epoch)

    def train_one_epoch(self):
        self.model.train()
        meters = {
            'batch_time': AverageMeter(), 'loss': AverageMeter(),
            'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(),
            'f1_score': AverageMeter()
        }
        end_time = time.time()
        for inx, (img, mask, label, clinical) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            img, label, clinical, mask = img.to(self.device), label.to(self.device), clinical.to(self.device), mask.to(
                self.device)
            cls = self.model(img)[-1]
            pred = torch.sigmoid(cls)
            loss = self.loss_function(pred, label)
            # dice_loss = self.dice_loss(seg, mask)
            # loss = self.loss_weight[0] * cls_loss + self.loss_weight[1] * dice_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc, precision_val, recall_val, f1_score_val, dice = self.calculate_metrics(pred, label)
            self.update_meters(
                [meters['loss'], meters['accuracy'], meters['precision'], meters['recall'], meters['f1_score']],
                [loss.item(), acc, precision_val, recall_val, f1_score_val])
            meters['batch_time'].update(time.time() - end_time)
            end_time = time.time()
            if (inx + 1) % self.args.log_interval == 0:
                self.print_metrics(meters, prefix=f'Epoch: [{self.epoch}][{inx + 1}/{len(self.train_loader)}]')

        metrics = {
            'Accuracy': meters['accuracy'].avg, 'Precision': meters['precision'].avg,
            'Recall': meters['recall'].avg, 'F1_Score': meters['f1_score'].avg,
            'Loss': meters['loss'].avg
        }
        self.log_metrics_to_tensorboard(metrics, self.epoch, stage_prefix='Train')
        self.log_metrics_to_tensorboard({'lr':self.optimizer.param_groups[0]['lr']}, self.epoch)

    def evaluate(self):
        self.model.eval()  # 切换模型到评估模式
        meters = {
            'loss': AverageMeter(),
            'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(),
            'f1_score': AverageMeter()
        }

        with torch.no_grad():  # 禁用梯度计算
            for inx, (img, mask, label, clinical) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                img, label, clinical, mask = img.to(self.device), label.to(self.device), clinical.to(
                    self.device), mask.to(self.device)
                cls = self.model(img)[-1]
                pred = torch.sigmoid(cls)

                loss_val = self.loss_function(pred, label)
                # dice_loss_val = self.dice_loss(seg, mask)
                # total_loss_val = self.loss_weight[0] * cls_loss_val + self.loss_weight[1] * dice_loss_val

                acc, precision_val, recall_val, f1_score_val, dice_val = self.calculate_metrics(pred, label)

                self.update_meters(
                    [meters['loss'], meters['accuracy'], meters['precision'], meters['recall'],
                     meters['f1_score']],
                    [loss_val.item(), acc, precision_val, recall_val, f1_score_val]
                )

        # 更新学习率调度器
        self.scheduler.step(meters['loss'].avg)
        # 记录性能指标到TensorBoard
        metrics = {
            'Accuracy': meters['accuracy'].avg, 'Precision': meters['precision'].avg,
            'Recall': meters['recall'].avg, 'F1_Score': meters['f1_score'].avg,
            'Loss': meters['loss'].avg
        }
        self.log_metrics_to_tensorboard(metrics, self.epoch, stage_prefix='Val')

        # 检查并保存最佳模型
        if meters['accuracy'].avg > self.best_acc:
            self.best_acc = meters['accuracy'].avg
            # 保存模型检查点
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_acc': self.best_acc,
            }, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
            print(f"New best model saved at epoch {self.epoch} with accuracy: {self.best_acc:.4f}")
        self.print_metrics(meters, prefix=f'Epoch(Val): [{self.epoch}][{inx + 1}/{len(self.train_loader)}]')

        if self.epoch % self.args.save_epoch == 0:
            checkpoint = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_acc': meters['accuracy'].avg,
                    'num_params': self.num_params
                }
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
            print(f"New checkpoint saved at epoch {self.epoch} with accuracy: {meters['accuracy'].avg:.4f}")

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    model = generate_model(model_depth=args.rd, n_classes=args.num_classes, dropout_rate=args.dropout)

    # dropout_rate = 0.8  # Dropout概率，一般设置在0.3到0.5之间
    # num_features = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Dropout(dropout_rate),
    #     nn.Linear(num_features, args.num_classes-1)  # num_classes为您的数据集类别数
    # )

    # if torch.cuda.device_count() > 1:
    #     model = DataParallel(model)
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # data
    with open('configs/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data_dir = dataset['data_dir']
    # infos_name = dataset['infos_name']
    # filter_volume = dataset['filter_volume']
    # train_info, val_info = split_data(data_dir, infos_name, filter_volume, rate=0.8)
    with open(os.path.join(data_dir, 'train_clinical_infos.json'), 'r', encoding='utf-8') as f:
        train_info = json.load(f)
    with open(os.path.join(data_dir, 'val_clinical_infos.json'), 'r', encoding='utf-8') as f:
        val_info = json.load(f)
    train_loader = my_dataloader(data_dir,
                                      train_info,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
    val_loader = my_dataloader(data_dir,
                                     val_info,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)
    summaryWriter = None
    if args.phase == 'train':

        log_path = makedirs(os.path.join(path, 'logs'))
        model_path = makedirs(os.path.join(path, 'models'))
        args.log_dir = log_path
        args.save_dir = model_path
        summaryWriter = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model,
                      optimizer,
                      device,
                      train_loader,
                      val_loader,
                      scheduler,
                      args,
                      summaryWriter)
    trainer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--rd', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    # parser.add_argument('--input-path', type=str, default='/home/wangchangmiao/kidney/data/')
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--dropout', type=float, default=0)

    opt = parser.parse_args()
    args_dict = vars(opt)
    now = time.strftime('%y%m%d%H%M', time.localtime())
    path = None
    if opt.phase == 'train':
        if not os.path.exists(f'./results/{now}'):
            os.makedirs(f'./results/{now}')
        path = f'./results/{now}'
        with open(os.path.join(path, 'train_config.json'), 'w') as fp:
            json.dump(args_dict, fp, indent=4)
        print(f"Training configuration saved to {now}")
    print(args_dict)

    main(opt, path)