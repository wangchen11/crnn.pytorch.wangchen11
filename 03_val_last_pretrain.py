import os
import json
import torch
import crnn.trainer

def lastTrain(path: str) -> str:
    files = os.listdir(path)
    lastFile = None
    lastMTime = 0
    for item in files:
        if not item.endswith(".pth"):
            continue
        file = f"{path}/{item}"
        mtime = os.path.getmtime(file)
        if (mtime > lastMTime):
            lastMTime = mtime
            lastFile = file
        pass
    if lastFile == None:
        raise Exception(f"no last **.pth found in `{path}`")
    return lastFile

trainerOpt = crnn.trainer.TrainerOpt()
trainerOpt.batchSize = 64
trainerOpt.adadelta = True
trainerOpt.cuda = torch.cuda.is_available()
trainerOpt.trainRoot = "out/db/pretrain/train"
trainerOpt.valRoot = "out/db/pretrain/val"
trainerOpt.pretrained = lastTrain(trainerOpt.exprDir) # out/expr/**.pth
# print(f"trainerOpt:{json.dumps(trainerOpt.__dict__, indent=2)}")
trainer = crnn.trainer.Trainer(trainerOpt)
trainer.val(10000)