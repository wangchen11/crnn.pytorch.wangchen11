import random

class TextGeneratorOpt():
    def __init__(self) -> None:
        pass

class TextGenerator():
    def __init__(self, opt: TextGeneratorOpt) -> None:
        self.opt: TextGenerator = opt
        pass
    
    def next(self) -> str:
        alphaNet = "abcdefghijklmnopqrstuvwxyz"
        lable = ""
        for j in range(random.randint(1, 9)):
            index = random.randint(0, len(alphaNet) - 1)
            lable = f"{lable}{alphaNet[index]}"
        return lable
