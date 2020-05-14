from Ge_search import init, genetic

# hyper-parameters
num_classes = 10
num_epochs = 30 # 40
generations = 1 # 代数 20
batch_size = 128
learning_rate = 0.025

num_individuals = 1 # 种群大小 5
num_phases = 3 # phase数量
num_nodes = 4 # 每个phase节点个数


if __name__ == '__main__':
    individuals = init(num_individuals, num_phases, num_nodes)
    individuals = genetic(individuals, num_classes, num_epochs, num_individuals, generations, batch_size, learning_rate)
    print('经过%o代进化，种群个体进化为：\n' % generations)
    for individual in individuals:
        print('个体的结构为{}，结构大小为：{}MB，其精度为{}%。\n'.format(individual.structure, individual.size, individual.accuracy))
