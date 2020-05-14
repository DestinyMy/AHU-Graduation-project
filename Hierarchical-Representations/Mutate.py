import numpy as np


def mutate(genotype, raw_operations):
    id = np.random.randint(0, len(genotype)) # 选择随机操作进行变异
    genotype[id] = np.random.choice(raw_operations, 1)[0]
    return genotype
