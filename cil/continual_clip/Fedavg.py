import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import copy
import os
import random
from collections import defaultdict
import math
from . import utils
from continual_clip.sampling import sample_iid, sample_noniid
# 评估阶段需要构造跨任务的文本 tokens
from continual_clip.utils import get_class_ids_per_task, get_class_names
import clip


class FedAvgTrainer:
    """FedAvg训练器类，实现CLIP模型的联邦平均训练"""

    def __init__(self, cfg, global_model, client_model, train_dataset, eval_dataset, task_id, texts, classes_names=None,
                 prev_mean_acc_history=None):
        """
        初始化FedAvg训练器

        参数:
            cfg: 配置对象，包含训练参数
            global_model: 全局模型
            client_model: 客户端模型副本
            train_dataset: 训练数据集
            task_id: 当前任务ID
            texts: 文本tokens
        """
        self.cfg = cfg
        self.global_model = global_model
        self.client_model = client_model
        self.train_dataset = train_dataset
        self.task_id = task_id
        self.eval_dataset = eval_dataset  # 新增: 用于本地评估
        self.texts = texts  # 当前任务的文本 tokens
        self.classes_names = classes_names  # 全部类别名称 (可为 None，当不需要本地评估时)

        # 联邦学习参数
        self.iid = getattr(cfg, "iid", True)
        self.num_clients = getattr(cfg, "num_clients", 5)
        self.com_rounds = getattr(cfg, "com", 10)
        self.client_epochs = getattr(cfg, "client_epochs", 5)

        # 冻结模型选项
        self.freeze_image_encoder = getattr(cfg, "freeze_image_encoder", False)
        self.freeze_text_encoder = getattr(cfg, "freeze_text_encoder", False)

        # 训练稳定性参数
        self.gradient_clip_norm = getattr(cfg, "gradient_clip_norm", 1.0)

        # 跟踪训练进度的变量
        self.current_round = 0

        # 跟踪指标
        self.metrics = defaultdict(list)
        self.mean_acc_history = prev_mean_acc_history.copy() if prev_mean_acc_history else []

        # 创建数据划分 (加固逻辑)
        # 1. 首先获取当前任务的数据集
        current_task_dataset = train_dataset[task_id:task_id + 1]

        # 2. 在当前任务数据集上进行采样
        if self.iid:
            self.dict_users = sample_iid(current_task_dataset, self.num_clients)
        else:
            self.dict_users = sample_noniid(current_task_dataset, self.num_clients)

        # 3. 基于当前任务数据集创建客户端加载器
        self.clients_loaders = []
        self.client_sizes = []
        for uid in range(self.num_clients):
            client_indices = list(self.dict_users[uid])
            client_subset = Subset(current_task_dataset, client_indices)
            client_loader = DataLoader(
                client_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=getattr(cfg, "num_workers", 4),
                drop_last=True
            )
            self.clients_loaders.append(client_loader)
            self.client_sizes.append(len(client_subset))

        # --- 新增：一次性为所有任务的测试集进行客户端划分 ---
        if self.eval_dataset is not None:
            self.dict_users_test_all_tasks = []
            for t in range(len(self.eval_dataset)):  # 假设 eval_dataset 是一个可以获取总任务数的对象
                task_dataset = self.eval_dataset[t:t + 1]
                if self.iid:
                    self.dict_users_test_all_tasks.append(sample_iid(task_dataset, self.num_clients))
                else:
                    self.dict_users_test_all_tasks.append(sample_noniid(task_dataset, self.num_clients))
        # ---------------------------------------------------

        # 收集可训练参数名称
        self.exclude_params_name = []
        self.trainable_param_names = []

        # 如果启用了冻结选项，应用于模型
        if self.freeze_image_encoder or self.freeze_text_encoder:
            self._apply_freezing(self.global_model)
            self._apply_freezing(self.client_model)

        # 收集可训练参数名称
        # for name, param in self.global_model.named_parameters():
        #     if param.requires_grad and name not in self.exclude_params_name:
        #         self.trainable_param_names.append(name)
        # 输出可训练参数数量
        total_params = sum(p.numel() for p in self.global_model.parameters())
        trainable_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params}")
        print(f"FedAvg 可训练参数: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")

        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                self.trainable_param_names.append(name)


    def _apply_freezing(self, model):
        """应用冻结策略到模型"""
        # 记录冻结前的可训练参数数量
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 冻结视觉编码器
        if self.freeze_image_encoder:
            if hasattr(model, "visual"):
                for name, param in model.visual.named_parameters():
                    param.requires_grad = False
                print(f"已冻结视觉编码器参数 - 模型路径: model.visual")
            elif hasattr(model, "vision_model"):
                for name, param in model.vision_model.named_parameters():
                    param.requires_grad = False
                print(f"已冻结视觉编码器参数 - 模型路径: model.vision_model")

        # 冻结文本编码器
        if self.freeze_text_encoder:
            if hasattr(model, "transformer"):
                for name, param in model.transformer.named_parameters():
                    param.requires_grad = False
                print(f"已冻结文本编码器参数 - 模型路径: model.transformer")
            elif hasattr(model, "text_model"):
                for name, param in model.text_model.named_parameters():
                    param.requires_grad = False
                print(f"已冻结文本编码器参数 - 模型路径: model.text_model")

        # 记录冻结后的可训练参数数量
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 输出冻结的参数量
        if trainable_before != trainable_after:
            frozen_params = trainable_before - trainable_after
            print(f"剩余可训练参数占比: {trainable_after / (trainable_before + frozen_params) * 100:.2f}%")

    def _train_client(self, client_id, train_loader):
        """
        训练单个客户端模型

        参数:
            client_id: 客户端ID
            train_loader: 客户端数据加载器
        """
        self.client_model = self.client_model.float()
        self.client_model.train()

        # 训练参数
        params = []
        for name, param in self.client_model.named_parameters():
            if param.requires_grad and name not in self.exclude_params_name:
                params.append(param)

        # 优化器
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=getattr(self.cfg, "weight_decay", 0.01))

        scaler = torch.cuda.amp.GradScaler()

        total_iterations = len(train_loader) * self.client_epochs
        scheduler = utils.cosine_lr(optimizer, self.cfg.lr, 30, total_iterations)
        train_iter = iter(train_loader)
        progress_bar = tqdm(range(total_iterations), desc=f"客户端 {client_id} 训练 (任务 {self.task_id})")

        # 跟踪分类准确率以监控训练进展
        running_accuracy = 0.0
        running_loss = 0.0

        # 为当前客户端收集的指标
        client_metrics = defaultdict(list)

        # 获取期望的批次大小
        expected_batch_size = self.cfg.batch_size

        for iteration in range(total_iterations):
            # 获取数据
            try:
                inputs, targets, task_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            # 标签偏移处理 (加固逻辑)
            # print("标签处理前的 targets:", targets)
            if self.cfg.scenario == "class" and hasattr(self.cfg, "increment"):
                targets -= self.task_id * self.cfg.increment
            # print("标签处理后的 targets:", targets)
            # 确保标签非负且有效
            # assert torch.all(targets >= 0), f"负标签在偏移 {shift} 后被检测到"

            # 数据转移到GPU
            inputs = inputs.to(self.cfg.device)
            targets = targets.to(self.cfg.device).long()  # 确保 targets 为 long 类型

            # 混合精度训练
            with torch.cuda.amp.autocast():
                # 当前模型前向传播
                if self.cfg.use_moe or self.cfg.use_MoE_Adapters:
                    logits_per_image, _, aux_loss = self.client_model(inputs, self.texts, self.task_id, is_train=True)
                    # print("1")
                    # print("aux_loss:", aux_loss)
                else:
                    logits_per_image, _ = self.client_model(inputs, self.texts, self.task_id, is_train=True)

                # 计算交叉熵损失
                loss = F.cross_entropy(logits_per_image, targets, label_smoothing=getattr(self.cfg, "ls", 0.0))
                if self.cfg.use_moe or self.cfg.use_MoE_Adapters:
                    loss = loss + aux_loss

                # 记录当前批次的分类准确率
                with torch.no_grad():
                    _, predicted = torch.max(logits_per_image, 1)
                    batch_accuracy = (predicted == targets).float().mean().item()
                    running_accuracy = 0.9 * running_accuracy + 0.1 * batch_accuracy
                    running_loss = 0.9 * running_loss + 0.1 * loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # 使用scheduler更新学习率
            scheduler(iteration)

            # 记录指标
            current_loss = loss.item()

            # 记录指标到客户端指标中
            client_metrics["loss"].append(current_loss)
            client_metrics["accuracy"].append(batch_accuracy)

            # 更新进度条
            progress_info = {
                'loss': f"{running_loss:.4f}",
                'acc': f"{running_accuracy:.4f}"
            }
            progress_bar.set_postfix(progress_info)
            progress_bar.update(1)

        progress_bar.close()

        # 计算平均指标
        avg_metrics = {
            "loss": np.mean(client_metrics["loss"]) if client_metrics["loss"] else 0.0,
            "accuracy": np.mean(client_metrics["accuracy"]) if client_metrics["accuracy"] else 0.0
        }

        print(f"[客户端 {client_id}, 任务 {self.task_id}] 已完成 {total_iterations} 次迭代")
        print(f"最终指标 - 损失: {avg_metrics['loss']:.6f}, 准确率: {avg_metrics['accuracy']:.6f}")

        # 将客户端指标汇总到全局指标
        for key, values in client_metrics.items():
            if values:  # 确保有值
                self.metrics[key].extend(values)

        # 返回最终指标
        return avg_metrics

    def train(self):
        """执行联邦平均训练过程"""
        self.global_model.train()

        # 主训练循环(通信轮次)
        for global_round in range(self.com_rounds):
            self.current_round = global_round  # 更新当前轮次

            print(f"\n=== 全局通信轮次 [{global_round + 1}/{self.com_rounds}] ===")

            # 客户端训练
            client_states = []
            client_metrics = []  # 收集所有客户端指标

            for client_id, client_loader in enumerate(self.clients_loaders):
                print(f"\n--- 客户端 {client_id + 1}/{len(self.clients_loaders)} 训练 ---")

                # 每次训练前加载最新全局参数
                self.client_model.load_state_dict(self.global_model.state_dict())

                # 执行客户端训练并获取指标
                client_metric = self._train_client(
                    client_id=client_id,
                    train_loader=client_loader,
                )

                # 确保指标不为0或NaN
                for key, value in client_metric.items():
                    if np.isnan(value) or np.isinf(value):
                        print(f"警告: 客户端 {client_id} 的 {key} 指标为 {value}，替换为0")
                        client_metric[key] = 0.0

                client_metrics.append(client_metric)

                # 收集可训练参数状态
                client_state = {
                    name: self.client_model.state_dict()[name]
                    for name in self.trainable_param_names
                }
                client_states.append(client_state)

            # 模型聚合(FedAvg)
            print("\n--- 聚合全局模型 ---")
            global_state = self.global_model.state_dict()

            # 计算聚合权重
            total_size = sum(self.client_sizes)
            client_weights = [size / total_size for size in self.client_sizes]

            # 聚合模型参数
            for key in self.trainable_param_names:
                if all(key in state for state in client_states):
                    weighted_sum = torch.zeros_like(global_state[key])
                    for i, state in enumerate(client_states):
                        param_tensor = state[key].to(global_state[key].device)
                        weighted_sum += client_weights[i] * param_tensor
                    global_state[key] = weighted_sum

            # 加载聚合后的参数
            self.global_model.load_state_dict(global_state)

            # 保存当前轮客户端状态, 供最终评估使用
            self.last_client_states = client_states

            # 聚合并打印训练指标 - 使用加权平均
            for metric_name in ["loss", "accuracy"]:
                values = [m[metric_name] for m in client_metrics]
                weighted_avg = sum(v * w for v, w in zip(values, client_weights))
                print(f"全局轮次 {global_round + 1} 平均{metric_name}: {weighted_avg:.6f}")

        # 评估客户端并记录
        mean_acc = self.evaluate_clients()

        return self.global_model, self.mean_acc_history

    # ------------------ 新增函数 ------------------
    def evaluate_clients(self):
        """在任务结束后对每个客户端执行累积测试集评估，并记录结果"""

        print("\n=== 本地模型评估 (Task {:.0f}) ===".format(self.task_id))
        from torch.utils.data import ConcatDataset, Subset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------- 如果提供了类别名称，则为所有已见类别构造 text tokens --------
        if self.classes_names is not None:
            class_ids_per_task = list(get_class_ids_per_task(self.cfg))
            seen_class_ids = []
            for t in range(self.task_id + 1):
                seen_class_ids.extend(class_ids_per_task[t])

            seen_class_names = get_class_names(self.classes_names, seen_class_ids)
            prompts = [self.cfg.prompt_template.format(c) for c in seen_class_names]
            texts_tokens = clip.tokenize(prompts).to(device)
        else:
            texts_tokens = self.texts  # 退化到当前任务 tokens

        client_accs = []  # 收集每个客户端准确率
        total_samples = []  # 记录每个客户端的样本数，便于调试

        # -------- 创建本地模型副本，避免在循环中重复创建 --------
        local_model = copy.deepcopy(self.global_model).to(device)

        # 针对每个客户端单独评估
        for client_id in range(self.num_clients):
            # -------- 构建累积任务的测试集 --------
            subsets = []
            for t in range(self.task_id + 1):
                task_dataset = self.eval_dataset[t:t + 1]

                # --- 使用预先划分好的固定测试集索引 ---
                dict_users_test = self.dict_users_test_all_tasks[t]
                client_indices = list(dict_users_test[client_id])
                if client_indices:  # 确保索引列表不为空
                    subsets.append(Subset(task_dataset, client_indices))

            if not subsets:
                print(f"[评估警告] 客户端 {client_id} 在任务 0-{self.task_id} 上没有测试数据，跳过评估。")
                client_accs.append(0.0)
                total_samples.append(0)
                continue

            concat_testset = ConcatDataset(subsets)
            test_loader = DataLoader(
                concat_testset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=getattr(self.cfg, "num_workers", 4),
                drop_last=True
            )

            # -------- 加载客户端权重 --------
            # 总是先恢复到全局模型的状态，以清除上一个客户端的个性化参数
            local_model.load_state_dict(self.global_model.state_dict())
            if hasattr(self, "last_client_states") and client_id < len(self.last_client_states):
                client_state = self.last_client_states[client_id]
                # 将保存在CPU的参数加载到当前设备
                client_state_on_device = {k: v.to(device) for k, v in client_state.items()}
                local_model.load_state_dict(client_state_on_device, strict=False)

            local_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets, _ in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # 直接使用构造的累积文本 tokens 计算 logits
                    outputs = local_model(inputs, texts_tokens, self.task_id, is_train=False)
                    # 有些实现返回 (logits_image, logits_text, ...)，仅取图像 logits
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    preds_local = logits.argmax(dim=1)
                    correct += (preds_local == targets).sum().item()
                    total += targets.size(0)

            acc = 100.0 * correct / total if total > 0 else 0.0
            client_accs.append(acc)
            total_samples.append(total)
            print(f"客户端 {client_id} | 本地测试集准确率: {acc:.2f}% （样本数: {total}）")

        # -------- 打印统计信息（先任务准确率，再各客户端） --------
        mean_acc = sum(client_accs) / len(client_accs) if client_accs else 0.0
        print(f"任务 {self.task_id} | 平均准确率: {mean_acc:.2f}%")

        for cid, (acc, ns) in enumerate(zip(client_accs, total_samples)):
            print(f"客户端 {cid} | 本地测试集准确率: {acc:.2f}% （样本数: {ns}）")

        # -------- 计算并记录任务平均准确率 --------
        mean_acc = sum(client_accs) / len(client_accs) if client_accs else 0.0
        print(f"任务 {self.task_id} | 客户端平均准确率: {mean_acc:.2f}%")

        # --- 使用内存中的历史记录计算 avg_acc ---
        prev_accs = self.mean_acc_history  # 之前任务的记录
        total_tasks_done = len(prev_accs) + 1
        avg_acc = (sum(prev_accs) + mean_acc) / total_tasks_done

        # 记录并更新历史
        self.mean_acc_history.append(mean_acc)

        # 将结果写入独立日志，只记录 acc 与累计 avg_acc
        try:
            import os, json
            dir_name = os.path.dirname(self.cfg.log_path)
            base_name = os.path.basename(self.cfg.log_path)
            client_log_path = os.path.join(dir_name, f"client_{base_name}")

            log_entry = {
                "task": self.task_id,
                "acc": round(mean_acc, 2),
                "client_acc": [round(a, 2) for a in client_accs],
                "avg_acc": round(avg_acc, 2)
            }
            with open(client_log_path, 'a+') as f:
                f.write(json.dumps(log_entry) + '\n')

            # 额外打印累计平均准确率
            print(f"任务 {self.task_id} | 累计平均准确率 (avg_acc): {avg_acc:.2f}%")
        except Exception as e:
            print(f"写日志时出错: {e}")
        return mean_acc


def fedavg_train(global_model, train_dataset, eval_dataset, cfg, text, client_model, task_id, classes_names=None,
                 prev_mean_acc_history=None):
    """
    FedAvg训练的入口函数，供models.py调用

    参数:
        global_model: 全局模型
        train_dataset: 训练数据集
        cfg: 配置对象
        text: 文本tokens
        task_id: 当前任务ID
        client_model: 客户端模型副本

    返回:
        训练后的全局模型
    """
    # 设置随机种子以确保可复现性
    if hasattr(cfg, "seed"):
        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 这一个设置是：

    # 输出当前FedAvg配置信息
    print("\n=== FedAvg训练配置 ===")
    print(f"联邦通信轮次: {getattr(cfg, 'com', 10)}")
    print(f"客户端本地训练轮次: {getattr(cfg, 'client_epochs', 5)}")
    print(f"学习率: {getattr(cfg, 'lr', 3e-5)}")

    # 创建FedAvg训练器
    trainer = FedAvgTrainer(
        cfg=cfg,
        global_model=global_model,
        client_model=client_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_id=task_id,
        texts=text,
        classes_names=classes_names,
        prev_mean_acc_history=prev_mean_acc_history
    )

    # 执行训练
    trained_model, mean_acc_history = trainer.train()

    return trained_model, mean_acc_history 