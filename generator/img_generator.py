import os
import random
from PIL import Image,ImageDraw,ImageFont

from generator.text_generator import TextGenerator, TextGeneratorOpt

class ImgGeneratorOpt():
    def __init__(self) -> None:
        self.textOpt: TextGeneratorOpt = TextGeneratorOpt()
        pass

class ImgGenerator():
    def __init__(self, opt: ImgGeneratorOpt) -> None:
        self.opt: ImgGeneratorOpt = opt
        self.textGenerator: TextGenerator = TextGenerator(opt.textOpt)
        pass
    
    def next(self) -> tuple[str, Image.Image]:
        text = self.textGenerator.next()
        img = self.genPic(text)
        return text, img
    
    def randomFont(self) -> any:
        font=ImageFont.truetype('assets/fonts/SIMHEI.TTF',19)
        return font
    
    # range(0, 255)
    def randomGray(self, baseGray: int = None) -> int:
        if(baseGray == None):
            return random.randint(0, 255)
        for i in range(10000):
            newGray = self.randomGray()
            if(abs(baseGray - newGray) >= 90):
                return newGray
            pass
        raise Exception("can not gen a good gray in 10000's loop")

    def genPic(self, text) -> Image.Image:
        font = self.randomFont()
        bgGray = self.randomGray()
        fontGray = self.randomGray(bgGray)
        bgColor = (bgGray, bgGray, bgGray)
        fontColor = (fontGray, fontGray, fontGray)
        size = (100, 32)
        image=Image.new('RGB', size, bgColor)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), text, font=font)
        x,y,x1,y1 = box
        w,h = size
        leftW = w - x1 
        leftH = h - y1
        # print(f"text:{text} box:{box} leftW:{leftW} leftH:{leftH}")
        draw.text((random.random() * leftW, random.random() * leftH), text, font=font, fill=fontColor)
        return image
