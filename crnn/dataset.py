import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from generator.img_generator import ImgGenerator, ImgGeneratorOpt


class CsvDataset(Dataset):

    def __init__(self, root: str, transform=None, target_transform=None):
        annotations_file = f"{root}/_db.csv"
        img_dir = f"{root}"
        self.img_labels = pd.read_csv(annotations_file, dtype={0:"string", 1: "string", 1: "string"}, encoding='UTF-8')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        imgUrl = self.img_labels.iloc[index, 0]
        label = str(self.img_labels.iloc[index, 1])
        # print(f"imgUrl:{imgUrl} lable:{label}")
        img_path = os.path.join(self.img_dir, imgUrl)
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        # print(f"label:{label} image:{image}")
        return (image, label)

class AutoGeneratorDataset(Dataset):
    def __init__(self, imgOpt: ImgGeneratorOpt, total: int = 64000, transform=None, target_transform=None):
        self.imgGenerator = ImgGenerator(imgOpt)
        self.total = total
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        label, image, prop = self.imgGenerator.next()
        image = image.convert('L')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(f"label:{label} image:{image}")
        return (image, label)

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - tail)
            tail_index = random_start + torch.range(0, tail - 1)
            index[n_batch * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = ResizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
