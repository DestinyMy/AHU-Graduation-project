import numpy as np

from Encode import encode
from network_model import Network
from Evaluate import evaluate

class Individual(object):
    def __init__(self, structure=None, model=None, accuracy=0, size=0):
        self.structure = structure
        self.model = model
        self.model_dict = None
        self.accuracy = accuracy
        self.size = size


# 初始化种群，包含num_individuals个体
def init(num_individuals=10, num_phases=3, num_nodes=4, num_classes=10):
    individuals = []
    for i in range(num_individuals):
        individual = Individual(structure=encode(num_phases, num_nodes))
        individual.model = Network(individual.structure, [(3, 32), (32, 128), (128, 128)], num_classes, (32, 32))
        individuals.append(individual)
    return individuals


def get_accuracy(i):
    return i.accuracy


def select(individuals, num_individuals):
    individuals.sort(key=get_accuracy, reverse=True)
    return individuals[:num_individuals]


# 使用拉马克进化思想进行种群更新，新的个体汇继承父代的特征，加快训练
# 算子包括：添加层0，删除层1，替换层2
def produce(individuals, num_classes):
    index_individual = np.random.randint(0, len(individuals))
    structure = individuals[index_individual].structure # 随机个体当父代
    # 首先根据随机许选择得到算子类别
    operator_id = np.random.randint(0,3)
    id1 = np.random.randint(0, len(structure))  # 随机选择一个phase
    id2 = np.random.randint(0, len(structure[id1]))  # 选择操作层的下标
    if operator_id == 0:  # 添加卷积层
        structure[id1].insert(id2, np.random.choice([0,1],id2+1).tolist()) # 在对应下标处添加层
        # 更新后面的层
        for i in range(id2+1, len(structure[id1])):
            structure[id1][i].insert(id2+1, 0)
            pass
    elif operator_id == 1: # 删除卷积层
        _remove = structure[id1][id2]
        structure[id1].remove(_remove)
        # 更新
        for i in range(id2, len(structure[id1])):
            _remove = structure[id1][i][id2+1]
            structure[id1][i].remove(_remove)
    else: # 替换卷积层
        _change_length = len(structure[id1][id2])
        _change = np.random.choice([0,1], _change_length).tolist()
        structure[id1][id2] = _change
    individual = Individual(structure=structure)
    individual.model = Network(individual.structure, [(3, 32), (32, 128), (128, 128)], num_classes, (32, 32))
    # 继承父代特征，加快训练
    pre_dict = individuals[index_individual].model_dict
    model_dict = individual.model.state_dict()
    pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    model_dict.update(pre_dict)
    individual.model_dict = pre_dict
    individuals.append(individual)


def lemonade(individuals, num_individuals, num_classes, num_epochs, generations, batch_size, learning_rate):
    for i in range(generations):
        for j in range(len(individuals)):
            if individuals[j].accuracy < 1:
                evaluate(individuals[j], num_epochs, batch_size, learning_rate)
        individuals = select(individuals, num_individuals)
        produce(individuals, num_classes)
    return individuals
