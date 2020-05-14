from NSGA import init, nsga

# parameters
generations = 5 # 代数 5
num_individuals = 5 # 种群大小 5
num_phases = 4 # phase数量
num_nodes = 4 # 每个phase节点个数


if __name__ == '__main__':
    individuals = init(num_individuals, num_phases, num_nodes)
    individuals = nsga(individuals, num_individuals, generations)
    print('经过{}代进化，种群个体进化为：'.format(generations))
    for individual in individuals:
        print('个体结构为{}，其精度为{}。\n'.format(individual.structure, individual.accuracy))
