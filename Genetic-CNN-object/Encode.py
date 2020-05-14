import numpy as np


# 对每个phase进行编码
def phase_encode(num_nodes):
    phase = []
    random_pool = [0, 1]
    bit_nums = int(num_nodes * (num_nodes + 1) * 0.5)
    bit_string = np.random.choice(random_pool, bit_nums, replace=True)
    idx = 0
    for i in range(num_nodes):
        phase.append([])
        for j in range(i+1):
            phase[i].append(bit_string[idx])
            idx += 1
    phase.append([1]) # 添加残差连接
    return phase


# 使用phase编码进行整合
def encode(num_phases, num_nodes):
    operation_encode = []
    for i in range(num_phases):
        operation_encode.append(phase_encode(num_nodes))
    return operation_encode
