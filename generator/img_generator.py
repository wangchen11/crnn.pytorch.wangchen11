import os
import random
from PIL import Image,ImageDraw,ImageFont

from generator.text_generator import TextGenerator, TextGeneratorOpt

class ImgGeneratorOpt():
    def __init__(self) -> None:
        self.fontDir = "assets/fonts/"
        self.textOpt: TextGeneratorOpt = TextGeneratorOpt()
        pass
    
class ItemProp():
    def __init__(self) -> None:
        self.text: str
        self.imgSize: tuple[int, int]
        self.fontSize: int
        self.fontPath: str
        self.bgColor: tuple[int, int, int]
        self.fontColor: tuple[int, int, int]
        pass

class ImgGenerator():
    def __init__(self, opt: ImgGeneratorOpt) -> None:
        self.opt: ImgGeneratorOpt = opt
        self.textGenerator: TextGenerator = TextGenerator(opt.textOpt)
        self.allFontsPath: list = findFonts(opt.fontDir)
        self.fonts = {}
        pass
    
    def next(self) -> tuple[str, Image.Image, ItemProp]:
        prop = self.genItemProp()
        img = self.genPic(prop)
        return prop.text, img, prop
    
    def getFont(self, prop: ItemProp) -> any:
        fontKey = f"{prop.fontPath}--{prop.fontSize}"
        if hasattr(self.fonts, fontKey):
            font = self.fonts[fontKey]
        else:
            # print(f"fontPath:{fontPath}  fontSize:{fontSize}")
            font=ImageFont.truetype(prop.fontPath, prop.fontSize)
            self.fonts[fontKey] = font
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

    def genPic(self, prop: ItemProp) -> Image.Image:
        font = self.getFont(prop)
        image=Image.new('RGB', prop.imgSize, prop.bgColor)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), prop.text, font=font)
        x,y,x1,y1 = box
        w,h = prop.imgSize
        leftW = w - x1 
        leftH = h - y1
        # print(f"text:{text} box:{box} leftW:{leftW} leftH:{leftH}")
        draw.text((random.random() * leftW, random.random() * leftH), prop.text, font=font, fill=prop.fontColor)
        return image
    
    def genItemProp(self) -> ItemProp:
        prop = ItemProp()
        prop.text = self.textGenerator.next()
        prop.imgSize = (100, 32)
        prop.fontSize = random.randint(12, 19)
        prop.fontPath = self.allFontsPath[int(random.random() * len(self.allFontsPath))]
        bgGray = self.randomGray()
        fontGray = self.randomGray(bgGray)
        prop.bgColor = (bgGray, bgGray, bgGray)
        prop.fontColor = (fontGray, fontGray, fontGray)
        return prop 


def findFonts(path: str) -> list:
    files = os.listdir(path)
    fontsPath = []
    for item in files:
        if not (item.endswith(".ttf") or item.endswith(".TTF")):
            continue
        fontsPath.append(f"{path}/{item}")
        pass
    return fontsPath
