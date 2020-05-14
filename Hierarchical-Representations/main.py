import time
from threading import Thread
import numpy as np

from Init import init, raw_operations, Individual
from Evaluate import evaluate
from Mutate import mutate

# hyper-parameters
num_classes = 10
num_epochs = 40 # 40
generations = 30 # 30
batch_size = 128
learning_rate = 0.025

num_individuals = 10 # 10
num_workers = 10 # 10


def get_accuracy(i):
    return i.accuracy


# 选出每代中最好的N个个体
def select(individuals, num_individuals):
    individuals.sort(key=get_accuracy, reverse=True)
    return individuals[:num_individuals]


# 评估线程
def eval(individual, num_classes, num_epochs, batch_size, learning_rate):
    evaluate(individual, num_classes, num_epochs, batch_size, learning_rate)


def worker(individuals): # 工作器
    for individual in individuals:
        if individual.accuracy < 1:
            t = Thread(target=eval, args=(individual, num_classes, num_epochs, batch_size, learning_rate,))
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
    print('经过%o代进化后，种群为：' % generations)
    for individual in individuals:
        print('个体结构大小为{}MB，精度为{:.4f}'.format(individual.size, individual.accuracy))


if __name__ == '__main__':
    main()
