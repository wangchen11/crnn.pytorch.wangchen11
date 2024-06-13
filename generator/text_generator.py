import os
import random

class TextGeneratorOpt():
    def __init__(self) -> None:
        pass

class TextGenerator():
    def __init__(self, opt: TextGeneratorOpt) -> None:
        self.opt: TextGenerator = opt
        with open("assets/fonts/texts.txt", encoding="UTF-8") as f:
            self.text = f.read()
        if (self.text == None):
            raise Exception("can not get texts")
        pass
    
    def next(self) -> str:
        texts = self.text
        lable = ""
        for j in range(random.randint(1, 5)):
            index = random.randint(0, len(texts) - 1)
            lable = f"{lable}{texts[index]}"
        return lable
