import numpy as np

from Mutate import mutate

raw_operations = ['1*1 convolution', '3*3 convolution','identity'] # 原始操作


class Individual(object):
    def __init__(self, genotype=None, accuracy=0):
        self.genotype = genotype
        self.accuracy = accuracy


def init(num_individuals):
    individuals = []
    genotype = np.random.choice(raw_operations, 7, replace=True).tolist() # 初始化一个基因型
    individuals.append(Individual(genotype=genotype))
    for i in range(1, num_individuals):
        genotype = mutate(genotype, raw_operations)
        individuals.append(Individual(genotype=genotype))
    return individuals
