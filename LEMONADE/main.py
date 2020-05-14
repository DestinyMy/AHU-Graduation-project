from LEMONADE import init, lemonade

# hyper-parameters
num_classes = 10
num_epochs = 40 # 40
generations = 1 # 代数 5
batch_size = 128
learning_rate = 0.025

num_individuals = 1 # 种群大小 5
num_phases = 3 # phase数量
num_nodes = 4 # 每个phase节点个数

if __name__ == '__main__':
    individuals = init(num_individuals, num_phases, num_nodes, num_classes)
    population = lemonade(individuals, num_individuals, num_classes, num_epochs, generations, batch_size, learning_rate)
    print('经过{}代后，种群进化为：\n'.format(generations))
    for individual in individuals:
        print('个体结构为：{}，其精度为：{}'.format(individual.structure, individual.accuracy))
