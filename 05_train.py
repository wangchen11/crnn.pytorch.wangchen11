import json

import crnn.trainer

trainerOpt = crnn.trainer.TrainerOpt()
trainerOpt.parseByArgs()
print(f"trainerOpt:{json.dumps(trainerOpt.__dict__, indent=2)}")
trainer = crnn.trainer.Trainer(trainerOpt)
trainer.trainLoop()