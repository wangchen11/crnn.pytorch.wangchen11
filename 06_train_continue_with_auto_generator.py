import json
import torch
import crnn.trainer
from crnn.utils import lastTrain
from generator.img_generator import ImgGenerator, ImgGeneratorOpt

trainerOpt = crnn.trainer.TrainerOpt()
trainerOpt.batchSize = 8
trainerOpt.adadelta = True
trainerOpt.cuda = torch.cuda.is_available()
# trainerOpt.trainRoot = "out/db/pretrain/train"
# trainerOpt.valRoot = "out/db/pretrain/val"
trainerOpt.trainImgOpt = ImgGeneratorOpt()
trainerOpt.valImgOpt = ImgGeneratorOpt()
trainerOpt.pretrained = lastTrain(trainerOpt.exprDir) # out/expr/**.pth
print(f"trainerOpt:{trainerOpt}")
trainer = crnn.trainer.Trainer(trainerOpt)
trainer.trainLoop()