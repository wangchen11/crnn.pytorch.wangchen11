from __future__ import print_function
from __future__ import division
import json

from src.trainer import Trainer, TrainerOpt


trainerOpt = TrainerOpt()
trainerOpt.parseByArgs()
print(f"trainerOpt:{json.dumps(trainerOpt.__dict__, indent=2)}")
trainer = Trainer(trainerOpt)
trainer.prepare()
