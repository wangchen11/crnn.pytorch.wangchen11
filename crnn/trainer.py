import json
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import crnn.utils as utils
from crnn.utils import StrLabelConverter, Averager

from torch.nn import CTCLoss
from torch.utils.data import Dataset

import argparse
import os

from crnn.crnn import CRNN
from crnn.dataset import CsvDataset, AlignCollate, ResizeNormalize

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
    trainLoader: torch.utils.data.DataLoader
    valLoader: torch.utils.data.DataLoader
    converter: StrLabelConverter
    criterion: CTCLoss
    crnn: CRNN
    lossAvg: Averager
    image: torch.FloatTensor
    text: torch.IntTensor
    length: torch.IntTensor
    
    def __init__(self, opt: TrainerOpt):
        random.seed(opt.manualSeed)
        np.random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        cudnn.benchmark = True

        print(f"torch version:{torch.__version__}")
        print(f"torch.cuda.is_available():{torch.cuda.is_available()}")

        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            
        os.makedirs(opt.exprDir, exist_ok=True)
        
        # if not opt.random_sample:
        #     sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
        #     shuffle = False
        # else:
        #     sampler = None
        #     shuffle = True

        sampler = None
        shuffle = True
        
        trainDataset = CsvDataset(opt.trainRoot)
        trainLoader = torch.utils.data.DataLoader(
            trainDataset, batch_size=opt.batchSize,
            shuffle=shuffle, sampler=sampler,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keepRatio))
        testDataset = CsvDataset(opt.valRoot, transform=ResizeNormalize((100, 32)))
        valLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=opt.batchSize)

        nc = 1
        
        converter = StrLabelConverter()
        criterion = CTCLoss()

        crnn = CRNN(opt.imgH, nc, opt.nh)
        crnn.weightsInit()
        
        if opt.cuda:
            # https://blog.csdn.net/anshiquanshu/article/details/122157157?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-122157157-blog-133892292.235%5Ev43%5Epc_blog_bottom_relevance_base1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-122157157-blog-133892292.235%5Ev43%5Epc_blog_bottom_relevance_base1&utm_relevant_index=3
            # gpu并行需要下面这句， 但是开启后会导致。使用cuda创建的模型在不使用cuda时无法加载。 但是大部分电脑都不支持多gpu。所以先屏蔽
            # self.crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
            pass
        
        if opt.pretrained:
            print('loading pretrained model from %s' % opt.pretrained)
            crnn.load_state_dict(torch.load(opt.pretrained))
        print(crnn)
        
        image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
        text = torch.IntTensor(opt.batchSize * 5)
        length = torch.IntTensor(opt.batchSize)
        
        if opt.cuda:
            crnn.cuda()
            image = image.cuda()
            criterion = criterion.cuda()
            
        # setup optimizer
        if opt.adam:
            optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                                betas=(opt.beta1, 0.999))
        elif opt.adadelta:
            optimizer = optim.Adadelta(crnn.parameters())
        else:
            optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

        self.opt: TrainerOpt = opt
        self.trainLoader: torch.utils.data.DataLoader = trainLoader
        self.valLoader: torch.utils.data.DataLoader = valLoader
        self.converter: StrLabelConverter = converter
        self.criterion: CTCLoss = criterion
        self.crnn: CRNN = crnn
        self.image: Variable = Variable(image)
        self.text: Variable = Variable(text)
        self.length: Variable = Variable(length)
        self.lossAvg = Averager()
        self.optimizer = optimizer
        pass
    
    def trainLoop(self):
        opt = self.opt
        trainLoader = self.trainLoader
        crnn = self.crnn
        lossAvg = self.lossAvg
        
        self.val()

        for epoch in range(opt.nepoch):
            trainIter = iter(self.trainLoader)
            i = 0
            while True:
                for p in crnn.parameters():
                    p.requires_grad = True
                    
                data = next(trainIter, None)
                if data == None:
                    break
                crnn.train()
                cost = self.trainBatch(data)
                lossAvg.add(cost)
                i += 1

                if i % opt.displayInterval == 1:
                    print('[%d/%d][%d/%d] Loss: %f' %
                        (epoch, opt.nepoch, i, len(trainLoader), lossAvg.val()))
                    lossAvg.reset()

                if i % opt.valInterval == 1:
                    self.val()

                # do checkpointing
                if i % opt.saveInterval == 1:
                    torch.save(
                        crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.exprDir, epoch, i))

        self.val()
        torch.save(crnn.state_dict(), f'{opt.exprDir}/crnn.pth')
        
    def trainBatch(self, data: tuple):
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        
        crnn = self.crnn
        criterion = self.criterion
        optimizer = self.optimizer
        image = self.image
        text = self.text
        length = self.length
        converter = self.converter
        
        # print(f"cpu_images:{cpu_images.shape}")
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        # print(f"cpu_texts:{cpu_texts}")
        # print(f"t:{t}")
        # print(f"l:{l}")
        # print(f"t.shape:{t.shape}")
        # print(f"l.shape:{l.shape}")
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        
        cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        return cost
    
    
    def val(self, max_iter = 20):
        opt = self.opt
        crnn = self.crnn
        criterion = self.criterion
        image = self.image
        text = self.text
        length = self.length
        converter = self.converter
        valLoader = self.valLoader
        
        print('Start val')
        for p in crnn.parameters():
            p.requires_grad = False

        crnn.eval()
        val_iter = iter(valLoader)

        i = 0
        n_correct = 0
        n_count = 0
        loss_avg = Averager()

        max_iter = min(max_iter, len(valLoader))
        for i in range(max_iter):
            data = next(val_iter)
            i += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            # print(f"preds.data:{preds.data}")
            # print(f"preds_size.data:{preds_size.data}")
            # print(f"preds.data.shape:{preds.data.shape}")
            # print(f"preds_size.data.shape:{preds_size.data.shape}")
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                n_count += 1
                if pred == target:
                    n_correct += 1

        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.nTestDisp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            if pred == gt:
                right = ""
            else:
                right = "×"
            print('%-20s => %-20s, gt: %-20s  %s' % (raw_pred, pred, gt, right))

        accuracy = n_correct / float(n_count)
        print('Test loss: %f, accuray: %f = %d/%d' % (loss_avg.val(), accuracy, n_correct, n_count))
        pass
    