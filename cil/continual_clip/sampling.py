import numpy as np
import torch
from torchvision import datasets, transforms


def sample_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = list(range(len(dataset)))  # 转换为列表
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)# 从all_idxs中随机选择num_items个元素，不放回
        all_idxs = list(set(all_idxs) - set(dict_users[i]))  # 转换为列表
    return dict_users


# 修改sample_noniid函数
import numpy as np
import torch
from torchvision import datasets, transforms

#设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def sample_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = list(range(len(dataset)))  # 转换为列表

    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def sample_noniid(dataset, num_users, beta=0.1, min_require_size=1):
    """
    基于 Dirichlet 分布划分数据，使得不同用户之间具有非 IID 特性。

    参数:
        dataset: 数据集对象，要求有属性 _y，存储所有样本的标签（NumPy 数组）。
        num_users: 用户数量。
        beta: Dirichlet 分布的参数，beta 越小，不同用户间数据分布越不均衡。
        min_require_size: 每个用户至少需要的样本数，默认设为 1。

    返回:
        dict_users: 一个字典，key 为用户编号，value 为该用户对应的样本下标（NumPy 数组）。
    """
    y = dataset._y
    n_samples = len(y)
    labels = np.unique(y)

    # 初始化用户数据索引映射
    dict_users = {}
    min_size = 0

    # 采用 while 循环，确保每个用户至少有 min_require_size 个样本
    while min_size < min_require_size:
        # 初始化每个用户的样本列表为空
        idx_batch = [[] for _ in range(num_users)]
        # 针对每个类别分别划分数据
        for k in labels:
            idx_k = np.where(y == k)[0]  # 获取当前类别的所有样本下标
            np.random.shuffle(idx_k)  # 打乱顺序
            # 根据 Dirichlet 分布生成概率分配
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            # 对于样本数超过单用户平均数的，继续分配；不足则比例置零
            proportions = np.array([p * (len(idx_j) < n_samples / num_users)
                                    for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()  # 归一化概率
            # 根据概率计算分割点
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # 将当前类别样本按照计算好的概率分给各个用户
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        # 检查所有用户的最小样本数是否满足要求
        min_size = min([len(idx_j) for idx_j in idx_batch])

    # 将列表转换为 NumPy 数组，并进行一次随机打乱
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        dict_users[i] = np.array(idx_batch[i], dtype='int64')

    #输出每个clients的样本数
    for i in range(num_users):
        print("client {} has {} samples".format(i, len(dict_users[i])))
    return dict_users
