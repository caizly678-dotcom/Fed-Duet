import numpy as np
import torch

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
    y = dataset._y
    n_samples = len(y)
    labels = np.unique(y)
    dict_users = {}
    min_size = 0

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in labels:
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < n_samples / num_users)
                                    for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()  # 归一化概率
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])


    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        dict_users[i] = np.array(idx_batch[i], dtype='int64')

    for i in range(num_users):
        print("client {} has {} samples".format(i, len(dict_users[i])))
    return dict_users
