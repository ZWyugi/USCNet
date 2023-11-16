# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings
warnings.filterwarnings("ignore")
import logging  # 引入logging模块
import os.path
import time
import os
import math
import argparse
import torch
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

class Logger:
    def __init__(self,mode='w'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.getcwd() + '/Logs/'
        log_name = log_path + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


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
        self.load_model()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.summaryWriter = summaryWriter

    def __call__(self):
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            self.epoch = epoch+1
            self.train_one_epoch()
            self.num_params = sum([param.nelement() for param in self.model.parameters()])
            self.scheduler.step()
            end = time.time()
            print("Epoch: {}, train time: {}".format(epoch, end - start))
            if epoch % 1 == 0:
                self.evaluate()

    def load_model(self):
        if self.args.MODEL_WEIGHT:

            model_dict = self.model.state_dict()
            pretrain = torch.load(self.args.MODEL_WEIGHT)
            pretrained_dict = {k: v for k, v in pretrain['state_dict'].items() if
                               k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

            print('load model weight success!')

    def evaluate(self):
        self.model.eval()
        total_step = 0
        per_epoch_loss = 0
        per_epoch_num_correct = 0
        with torch.no_grad():
            for inx, (x, mask, label) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                x = x.to(self.device)
                label = label.to(self.device)
                total_step += x.shape[0]
                pred = self.model(x)[-1]
                pred = F.softmax(pred, dim=-1)
                loss = self.loss_function(pred, label)
                per_epoch_loss += loss.item()
                pred_class = pred.argmax(dim=1)
                per_epoch_num_correct += torch.eq(pred_class, label).sum().item()
            test_acc = per_epoch_num_correct / total_step
            print(f'TEST: Epoch:{self.epoch}/{self.epochs}, Loss:{per_epoch_loss/(inx+1)}, acc:{test_acc}')
            self.summaryWriter.add_scalar("Loss/TEST", per_epoch_loss/len(self.test_loader), self.epoch)
            self.summaryWriter.add_scalar("acc/TEST", test_acc, self.epoch)


            if self.epoch % self.args.save_epoch == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': per_epoch_num_correct / total_step,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
                logger.logger.info('save model %d successed......\n'%self.epoch)

            if self.best_acc < test_acc:
                self.best_acc = test_acc
                # logger.logger.info('best model in %d epoch, train acc: %.3f \n' % (self.epoch, train_acc))
                # logger.logger.info('best model in %d epoch, validation acc: %.3f \n' % (epoch, val_acc))
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': self.best_acc,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
                logger.logger.info('save best model  successed......\n')

    def train_one_epoch(self):
        per_epoch_loss = 0
        total_step = 0
        num_correct = 0
        self.model.train()
        for inx, (x, mask, label) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            x = x.to(self.device)
            label = label.to(self.device)
            pred = self.model(x)[-1]
            pred = F.softmax(pred, dim=-1)
            loss = self.loss_function(pred, label)
            per_epoch_loss += loss.item()
            pred_class = pred.argmax(dim=1)

            loss = self.loss_function(pred, label)
            per_epoch_loss += loss.item()
            total_step += x.shape[0]
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(f'logits:{logits}, pred:{pred}, label:{label}')
            num_correct += torch.eq(pred_class, label).sum().item()
            # if inx % 5 == 0:
            # print(f'iters:{inx}/{len(self.train_loader)}, Loss:{loss.item()}, acc:{num_correct/total_step}')
        self.summaryWriter.add_scalar("Loss/TRAIN", per_epoch_loss / len(self.train_loader), self.epoch)
        self.summaryWriter.add_scalar("acc/TRAIN", num_correct/total_step, self.epoch)
        print(f'train epoch:{self.epoch}/{self.epochs}, Loss:{per_epoch_loss / len(self.train_loader)}, acc:{num_correct / total_step}')

def main(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    # model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=args.num_classes)
    model = generate_model(model_depth=18, n_classes=args.num_classes)
    # model = models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features  # 获取低级特征维度
    # model.fc = nn.Linear(num_ftrs, args.num_classes)  # 替换新的输出层

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
    # if not os.path.exists(data_dir):
    #     data_dir = '/home/wangchangmiao/kidney/data/data'
    data_dir = args.input_path
    train_infos, val_infos = split_data(data_dir)
    train_loader = my_dataloader(data_dir, train_infos, batch_size=args.batch_size, input_size=args.input_size)
    val_loader = my_dataloader(data_dir, val_infos, batch_size=args.batch_size, input_size=args.input_size)
    logger.logger.info('start training......\n')
    summaryWriter = SummaryWriter(log_dir=args.log_dir)
    # train_writer = SummaryWriter(os.path.join(summary_dir, 'train'), flush_secs=2)
    # test_writer = SummaryWriter(os.path.join(summary_dir, 'test'), flush_secs=2)
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
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='./Logs')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_size', type=str, default="128, 128, 128")
    parser.add_argument('--input_path', type=str, default='/home/wangchangmiao/kidney/data/data')
    parser.add_argument('--MODEL_WEIGHT', type=str, default=None)

    opt = parser.parse_args()
    args_dict = vars(opt)
    now = time.strftime('%y%m%d%H%M', time.localtime())
    with open(f'./configs/training_config_{now}.json', 'w') as fp:
        json.dump(args_dict, fp, indent=4)
    print(f"Training configuration saved to training_config_{now}.json")
    print(args_dict)

    logger = Logger()
    main(opt, logger)