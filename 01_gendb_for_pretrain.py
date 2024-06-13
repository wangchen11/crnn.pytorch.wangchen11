
from generator.db_generator import DbGenerator, DbGeneratorOpt

opt = DbGeneratorOpt()
opt.dir = "out/db/pretrain/train/"
generator = DbGenerator(opt)
generator.begin()
generator.genNextN(10000)
generator.end()

opt = DbGeneratorOpt()
opt.dir = "out/db/pretrain/val/"
generator = DbGenerator(opt)
generator.begin()
generator.genNextN(1000)
generator.end()