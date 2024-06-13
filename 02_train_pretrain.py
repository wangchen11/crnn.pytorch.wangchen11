import json

import torch

import crnn.trainer

trainerOpt = crnn.trainer.TrainerOpt()
trainerOpt.batchSize = 2
trainerOpt.adadelta = True
trainerOpt.cuda = torch.cuda.is_available()
trainerOpt.trainRoot = "out/db/pretrain/train"
trainerOpt.valRoot = "out/db/pretrain/val"
print(f"trainerOpt:{json.dumps(trainerOpt.__dict__, indent=2)}")
trainer = crnn.trainer.Trainer(trainerOpt)
trainer.trainLoop()