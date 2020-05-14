import numpy as np

from Encode import encode
from train import train

class Individual(object):
    def __init__(self, structure=None, accuracy=0):
        self.structure = structure
        self.accuracy = accuracy


# 初始化种群，包含num_individuals个体
def init(num_individuals=10, num_phases=3, num_nodes=4):
    individuals = []
    for i in range(num_individuals):
        individual = Individual(structure=encode(num_phases, num_nodes))
        individuals.append(individual)
    return individuals


# 轮盘赌选择个体
def select(individuals, num_individuals):
    new_individuals = [] # 通过轮盘赌产生的种群
    accuracy = [] # 取出精度
    rate = [] # 得到概率
    # 取出所有个体精度
    for individual in individuals:
        accuracy.append(individual.accuracy)
    all = sum(accuracy)
    for a in accuracy:
        rate.append(a/all)
    accumulate = rate # 计算累计概率
    for i in range(1, len(rate)):
        accumulate[i] = accumulate[i-1] + rate[i]
    accumulate[-1] = 1.0
    for i in range(num_individuals):
        rate = np.random.rand()
        for j in range(len(accumulate)):
            if rate < accumulate[j]:
                new_individuals.append(individuals[j])
                break
    return new_individuals


# 随机选择两个个体进行交叉
def crossover(individuals):
    id1 = id2 = 0
    while True:
        id1 = np.random.randint(0, len(individuals))
        id2 = np.random.randint(0, len(individuals))
        if id1 != id2:
            break
    structure1 = individuals[id1].structure
    structure2 = individuals[id2].structure
    structure = structure1
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            for k in range(len(structure[i][j])):
                if structure1[i][j][k] == structure2[i][j][k]:
                    structure[i][j][k] = structure1[i][j][k]
                elif np.random.rand() >= 0.5:
                    structure[i][j][k] = structure1[i][j][k]
                else:
                    structure[i][j][k] = structure2[i][j][k]
    individual = Individual(structure=structure)
    individuals.append(individual)


# 随机选择一个个体进行一位变异
def mutate(individuals):
    index = np.random.randint(0, len(individuals)) # 随即得到一个个体的编号
    structure = individuals[index].structure
    id1 = np.random.randint(0, len(structure))
    id2 = np.random.randint(0,len(structure[id1]))
    id3 = np.random.randint(0, len(structure[id1][id2]))
    structure[id1][id2][id3] = 0 if structure[id1][id2][id3] == 1 else 1
    individual = Individual(structure=structure)  # 复制原有选中个体
    individuals.append(individual)


def genetic(individuals, num_individuals, generations):
    for i in range(generations):
        for j in range(len(individuals)):
            if individuals[j].accuracy < 1: # 该个体之前训练过就不训练
                train(individuals[j])
        individuals = select(individuals, num_individuals)
        crossover(individuals)
        mutate(individuals)
    return individuals
