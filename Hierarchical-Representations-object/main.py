import time
from threading import Thread
import numpy as np

from Init import init, raw_operations, Individual
from Mutate import mutate
from train import train

# hyper-parameters
generations = 5 # 5
num_individuals = 5 # 5
num_workers = 5 # 5


def get_accuracy(i):
    return i.accuracy


# 选出每代中最好的N个个体
def select(individuals, num_individuals):
    individuals.sort(key=get_accuracy, reverse=True)
    return individuals[:num_individuals]


def train_(individual):
    train(individual)


def worker(individuals): # 工作器
    for individual in individuals:
        if individual.accuracy < 1:
            t = Thread(target=train_, args=(individual,))
            t.start()
            t.join()
    time.sleep(0.1)


def hierarchical_search(generations, individuals, num_workers):
    for i in range(generations): # 控制器
        index = np.random.randint(0, len(individuals))
        individuals.append(Individual(genotype=mutate(individuals[index].genotype, raw_operations)))
        time.sleep(0.1)
    for j in range(num_workers):
        t = Thread(target=worker, args=(individuals,))
        t.start()


def main():
    individuals = init(num_individuals)

    t = Thread(target=hierarchical_search, args=(generations, individuals, num_workers,))
    t.start()
    t.join()

    individuals = select(individuals, num_individuals)
    print('经过{}代进化后，种群为：'.format(generations))
    for individual in individuals:
        print('个体结构大小为{}MB，精度为{:.4f}'.format(individual.size, individual.accuracy))


if __name__ == '__main__':
    main()
