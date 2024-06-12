import json
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np

import argparse
import os

class TrainerOpt:
    def __init__(self):
        self.trainRoot: str = "out/db/train/"
        self.valRoot: str  = "out/db/val/"
        self.workers: int = 0
        self.batchSize: int = 64
        self.imgH: int = 32
        self.imgW: int = 100
        self.nh: int = 256
        self.nepoch: int = 600
        self.cuda: bool = False
        self.ngpu: int = 1
        self.pretrained: str = None
        self.alphabet: str = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.exprDir: str = "out/expr"
        self.displayInterval: int = 50
        self.nTestDisp: int = 64
        self.valInterval: int = 1000
        self.saveInterval: int = 1000
        self.lr: float=0.01
        self.beta1: float=0.5
        self.adam: bool = False
        self.adadelta: bool = False
        self.keepRatio: bool = False
        self.manualSeed: int = 1234
        self.randomSample: bool = False
        pass
    
    def parseByArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--trainRoot', default=self.trainRoot, help='path to dataset')
        parser.add_argument('--valRoot', default=self.valRoot, help='path to dataset')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=self.workers)
        parser.add_argument('--batchSize', type=int, default=self.batchSize, help='input batch size')
        parser.add_argument('--imgH', type=int, default=self.imgH, help='the height of the input image to network')
        parser.add_argument('--imgW', type=int, default=self.imgW, help='the width of the input image to network')
        parser.add_argument('--nh', type=int, default=self.nh, help='size of the lstm hidden state')
        parser.add_argument('--nepoch', type=int, default=self.nepoch, help='number of epochs to train for')
        # TODO(meijieru): epoch -> iter
        parser.add_argument('--cuda', action='store_true', help='enables cuda')
        parser.add_argument('--ngpu', type=int, default=self.ngpu, help='number of GPUs to use')
        parser.add_argument('--pretrained', help="path to pretrained model (to continue training)")
        parser.add_argument('--alphabet', type=str, default=self.alphabet)
        parser.add_argument('--exprDir', default=self.exprDir, help='Where to store samples and models')
        parser.add_argument('--displayInterval', type=int, default=self.displayInterval, help='Interval to be displayed')
        parser.add_argument('--nTestDisp', type=int, default=self.nTestDisp, help='Number of samples to display when test')
        parser.add_argument('--valInterval', type=int, default=self.valInterval, help='Interval to be displayed')
        parser.add_argument('--saveInterval', type=int, default=self.saveInterval, help='Interval to be displayed')
        parser.add_argument('--lr', type=float, default=self.lr, help='learning rate for Critic, not used by adadealta')
        parser.add_argument('--beta1', type=float, default=self.beta1, help='beta1 for adam. default=0.5')
        parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
        parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
        parser.add_argument('--keepRatio', action='store_true', help='whether to keep ratio for image resize')
        parser.add_argument('--manualSeed', type=int, default=self.manualSeed, help='reproduce experiemnt')
        parser.add_argument('--randomSample', action='store_true', help='whether to sample the dataset with random sampler')
        parser.parse_args(namespace=self)
        pass
    

class Trainer:
    opt: TrainerOpt
    def __init__(self, opt: TrainerOpt):
        self.opt = opt
        pass
    
    def prepare(self):
        opt = self.opt
        random.seed(opt.manualSeed)
        np.random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)

        cudnn.benchmark = True

        print(f"torch version:{torch.__version__}")
        print(f"torch.cuda.is_available():{torch.cuda.is_available()}")

        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        if not os.path.exists(opt.exprDir):
            os.makedirs(opt.exprDir)
        pass