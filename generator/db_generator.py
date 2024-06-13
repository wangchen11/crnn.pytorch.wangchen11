import os

from generator.img_generator import ImgGenerator, ImgGeneratorOpt

class DbGeneratorOpt():
    def __init__(self, imgOpt: ImgGeneratorOpt = ImgGeneratorOpt()) -> None:
        self.imgOpt: ImgGeneratorOpt = imgOpt
        self.dir: str = "out/db/gen/"
        pass

def cvsEncode(text:str) -> str:
    text.replace('"', '""')
    return f'"{text}"'

class DbGenerator():
    def __init__(self, opt: DbGeneratorOpt) -> None:
        self.opt: DbGeneratorOpt = opt
        self.imgGenerator: ImgGenerator = ImgGenerator(self.opt.imgOpt)
        self.dbFile = f"{self.opt.dir}/_db.csv"
        self.count = 0
        pass
    
    def clearDb(self):
        with open(self.dbFile, 'w', encoding="UTF-8") as f:
            f.write("")
        pass
    
    def appendDb(self, text: str):
        with open(self.dbFile, 'a', encoding="UTF-8") as f:
            f.write(text)
        pass
    
    def begin(self, ) -> None:
        os.makedirs(self.opt.dir, exist_ok=True)
        self.clearDb()
        self.appendDb("image,lable\n")
        pass
    
    def end(self, ) -> None:
        pass
    
    def genNext(self, ) -> None:
        self.count = self.count + 1
        lable, img = self.imgGenerator.next()
        imgPath = f"{'%06d' % self.count}.png"
        img.save(f"{self.opt.dir}/{imgPath}")
        self.appendDb(f"{cvsEncode(imgPath)},{cvsEncode(lable)}\n")
        pass
    
    def genNextN(self, count: int) -> None:
        for i in range(count):
            self.genNext()
        pass
