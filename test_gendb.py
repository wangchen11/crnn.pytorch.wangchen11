
from generator.db_generator import DbGenerator, DbGeneratorOpt

opt = DbGeneratorOpt()
opt.dir = "out/db/test/"
opt.imgOpt.textOpt.textsFile = "assets/fonts/texts_sample.txt"
generator = DbGenerator(opt)
generator.begin()
generator.genNextN(100)
generator.end()
