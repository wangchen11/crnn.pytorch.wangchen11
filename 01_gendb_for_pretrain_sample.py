
from generator.db_generator import DbGenerator, DbGeneratorOpt

opt = DbGeneratorOpt()
opt.dir = "out/db/pretrain/train/"
opt.imgOpt.textOpt.textsFile = "assets/fonts/texts_sample.txt"
generator = DbGenerator(opt)
generator.begin()
generator.genNextN(10000)
generator.end()

opt = DbGeneratorOpt()
opt.dir = "out/db/pretrain/val/"
opt.imgOpt.textOpt.textsFile = "assets/fonts/texts_sample.txt"
generator = DbGenerator(opt)
generator.begin()
generator.genNextN(1000)
generator.end()