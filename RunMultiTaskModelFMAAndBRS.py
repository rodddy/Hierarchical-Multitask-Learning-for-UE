import math
import torch
import torch.nn as nn
import torch.utils.data as Data
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import  precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import json
import Models.MoEMHADeepConvLSTMMultitask as MoEMHADeepConvLSTMMultitask

# 超参数设置
LR = 0.005
Epoch = 80
base_dir = os.path.dirname(os.getcwd())
num_classes_brs = 6  # 根据实际BRS分类类别调整
num_classes_fma = 1
BATCH_SIZE = 128
save_interval = 10

# 加载多任务数据
def load_multitask_data():
    # 加载特征数据
    train_data = np.load(os.path.join(base_dir, 'dataset/multitask/x_train_4.npy'))
    val_data = np.load(os.path.join(base_dir, 'dataset/multitask/x_test_4.npy'))
    
    # 加载FMA回归标签
    train_labels_fma = np.load(os.path.join(base_dir, 'dataset/multitask/y_train_1_FMA.npy'))
    val_labels_fma = np.load(os.path.join(base_dir, 'dataset/multitask/y_test_1_FMA.npy'))
    
    # 加载BRS分类标签
    train_labels_brs = np.load(os.path.join(base_dir, 'dataset/multitask/y_train_1_Brunnstrom.npy'))
    val_labels_brs = np.load(os.path.join(base_dir, 'dataset/multitask/y_test_1_Brunnstrom.npy'))

    return train_data, val_data, train_labels_fma, val_labels_fma, train_labels_brs, val_labels_brs

# 构建多任务数据加载器
def create_multitask_dataloaders():
    train_data, val_data, train_fma, val_fma, train_brs, val_brs = load_multitask_data()
    
    # 转换为tensor
    Xtrain = torch.tensor(train_data, dtype=torch.float32)
    Ytrain_fma = torch.tensor(train_fma, dtype=torch.float32)
    Ytrain_brs = torch.tensor(train_brs, dtype=torch.long)  # 分类任务用long类型
    
    Xval = torch.tensor(val_data, dtype=torch.float32)
    Yval_fma = torch.tensor(val_fma, dtype=torch.float32)
    Yval_brs = torch.tensor(val_brs, dtype=torch.long)
    
    # 创建数据集
    train_dataset = TensorDataset(Xtrain, Ytrain_fma, Ytrain_brs)
    val_dataset = TensorDataset(Xval, Yval_fma, Yval_brs)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# 学习率调整
def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1 **(epoch // 50))
    optimizer.param_groups[0]['lr'] = lr

# 初始化模型和损失函数
actionIndex = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
weightpaths = [os.path.join(base_dir, f'experiment/Models/CNNLSTMPRE/model_{i}.pth') for i in actionIndex]
moe_weight_paths =[os.path.join(base_dir, f'experiment/Models/CNNLSTMPRE/model_multitask_FMA.pth'), os.path.join(base_dir, f'experiment/Models/CNNLSTMPRE/model_multitask_Brunnstrom.pth')]
#moe_weight_paths = None
# 初始化多任务模型（假设模型支持多输出）
model = MoEMHADeepConvLSTMMultitask.MoEMHADeepConvLSTMMultitask(
    input_channels=12,
    hidden_dim=64,
    kernel_size=5,
    num_layers=2,
    weight_paths=weightpaths,
    moe_weight_paths=moe_weight_paths,
    output_dim_fma=num_classes_fma,
    output_dim_brs=num_classes_brs  # 新增BRS分类输出维度
)
model.cpu()

# 定义损失函数 - 分别使用回归和分类损失
criterion_fma = nn.SmoothL1Loss()  # FMA回归损失
criterion_brs = nn.CrossEntropyLoss()  # BRS分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

ratio_fma = 0.5
ratio_brs = 1 - ratio_fma

# 初始化TensorBoard
strDir = datetime.now().strftime("%Y%m%d%H%M%S")
writer = {
    'train_loss_fma': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/train_loss_fma'),
    'train_loss_brs': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/train_loss_brs'),
    'train_loss_total': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/train_loss_total'),
    'train_mse': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/train_mse'),
    'train_mae': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/train_mae'),
    'train_acc_brs': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/train_acc_brs'),
    'val_loss_fma': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/val_loss_fma'),
    'val_loss_brs': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/val_loss_brs'),
    'val_loss_total': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/val_loss_total'),
    'val_mse': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/val_mse'),
    'val_mae': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/val_mae'),
    'val_acc_brs': SummaryWriter(log_dir=f'runs/multitask_joint_{ratio_fma}_{ratio_brs}/{strDir}/val_acc_brs')
}

# 训练函数
def train(epoch, loss_weights=[ratio_fma, ratio_brs]):
    adjust_learning_rate(optimizer, epoch)
    model.train()
    
    total_loss = 0
    total_loss_fma = 0
    total_loss_brs = 0
    total_mse = 0
    total_mae = 0
    total_correct_brs = 0
    total_samples = 0
    
    for step, (x, y_fma, y_brs) in enumerate(train_loader):
        x, y_fma, y_brs = x.cpu(), y_fma.cpu(), y_brs.cpu()
        batch_size = x.size(0)
        total_samples += batch_size
        
        # 前向传播 - 获取两个任务的输出
        output_fma, output_brs = model(x)
        output_fma = output_fma.squeeze(1)
        
        # 计算各任务损失
        loss_fma = criterion_fma(output_fma, y_fma)
        loss_brs = criterion_brs(output_brs, y_brs)
        
        # 联合损失（带权重）
        loss = loss_weights[0] * loss_fma + loss_weights[1] * loss_brs
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item() * batch_size
        total_loss_fma += loss_fma.item() * batch_size
        total_loss_brs += loss_brs.item() * batch_size
        
        # 计算回归指标
        mse, mae = calculate_metrics(y_fma, output_fma)
        total_mse += mse * batch_size
        total_mae += mae * batch_size
        
        # 计算分类指标
        _, predicted = torch.max(output_brs.data, 1)
        total_correct_brs += (predicted == y_brs).sum().item()
    
    # 计算平均指标
    avg_loss = total_loss / total_samples
    avg_loss_fma = total_loss_fma / total_samples
    avg_loss_brs = total_loss_brs / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    acc_brs = total_correct_brs / total_samples
    
    # 打印并记录指标
    print(f'Train Epoch:{epoch} | Total Loss:{avg_loss:.4f} | FMA Loss:{avg_loss_fma:.4f} | BRS Loss:{avg_loss_brs:.4f} | '
          f'FMA MSE:{avg_mse:.4f} | FMA MAE:{avg_mae:.4f} | BRS Acc:{acc_brs:.4f}')
    
    # 写入TensorBoard
    writer['train_loss_total'].add_scalar('Data', avg_loss, epoch)
    writer['train_loss_fma'].add_scalar('Data', avg_loss_fma, epoch)
    writer['train_loss_brs'].add_scalar('Data', avg_loss_brs, epoch)
    writer['train_mse'].add_scalar('Data', avg_mse, epoch)
    writer['train_mae'].add_scalar('Data', avg_mae, epoch)
    writer['train_acc_brs'].add_scalar('Data', acc_brs, epoch)
    
    # 保存模型
    if (epoch + 1) % save_interval == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join('Models', 'checkpoints',
                        f'model_multitask_joint_epoch_{epoch + 1}_f2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
        print(f'Model saved at epoch {epoch + 1}')

# 验证函数
def val(epoch):
    model.eval()
    total_loss = 0
    total_loss_fma = 0
    total_loss_brs = 0
    total_samples = 0
    
    all_fma_preds = []
    all_fma_labels = []
    all_brs_preds = []
    all_brs_labels = []
    
    with torch.no_grad():
        for step, (x, y_fma, y_brs) in enumerate(val_loader):
            x, y_fma, y_brs = x.cpu(), y_fma.cpu(), y_brs.cpu()
            batch_size = x.size(0)
            total_samples += batch_size
            
            # 前向传播
            output_fma, output_brs = model(x)
            output_fma = output_fma.squeeze(1)
            
            # 计算损失
            loss_fma = criterion_fma(output_fma, y_fma)
            loss_brs = criterion_brs(output_brs, y_brs)
            loss = 0.5 * loss_fma + 0.5 * loss_brs  # 使用相同权重
            
            # 累计损失
            total_loss += loss.item() * batch_size
            total_loss_fma += loss_fma.item() * batch_size
            total_loss_brs += loss_brs.item() * batch_size
            
            # 收集预测结果和标签
            all_fma_preds.extend(output_fma.cpu().numpy())
            all_fma_labels.extend(y_fma.cpu().numpy())
            _, brs_pred = torch.max(output_brs.data, 1)
            all_brs_preds.extend(brs_pred.cpu().numpy())
            all_brs_labels.extend(y_brs.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    avg_loss_fma = total_loss_fma / total_samples
    avg_loss_brs = total_loss_brs / total_samples
    
    # 计算FMA回归指标
    mse = mean_squared_error(all_fma_labels, all_fma_preds)
    mae = mean_absolute_error(all_fma_labels, all_fma_preds)
    
    # 计算BRS分类指标
    acc_brs = accuracy_score(all_brs_labels, all_brs_preds)
    f1_brs = f1_score(all_brs_labels, all_brs_preds, average='weighted')
    
    # 打印指标
    print(f'Val Epoch:{epoch} | Total Loss:{avg_loss:.4f} | FMA Loss:{avg_loss_fma:.4f} | BRS Loss:{avg_loss_brs:.4f} | '
          f'FMA MSE:{mse:.4f} | FMA MAE:{mae:.4f} | BRS Acc:{acc_brs:.4f} | BRS F1:{f1_brs:.4f}')
    
    # 写入TensorBoard
    writer['val_loss_total'].add_scalar('Data', avg_loss, epoch)
    writer['val_loss_fma'].add_scalar('Data', avg_loss_fma, epoch)
    writer['val_loss_brs'].add_scalar('Data', avg_loss_brs, epoch)
    writer['val_mse'].add_scalar('Data', mse, epoch)
    writer['val_mae'].add_scalar('Data', mae, epoch)
    writer['val_acc_brs'].add_scalar('Data', acc_brs, epoch)
    if (epoch + 1) % save_interval == 0:
        cm =confusion_matrix(all_brs_labels, all_brs_preds)
        metrics = metrics_from_confusion(cm)
        file_log = f'fma_brs_log_{ratio_fma}_{ratio_brs}_f2.txt'
        append_confusion_matrix_with_timestamp(metrics, epoch, save_dir='Models',file_name=file_log)
        calculate_regression_metrics(all_fma_labels, all_fma_preds, model_name=f"multitask_fma_brs",save_dir='Models',file_name=file_log)

# 评估并保存指标
def eval_model(all_fma_labels, all_fma_preds, all_brs_labels, all_brs_preds):
    # 计算FMA回归指标
    calculate_regression_metrics(
        all_fma_labels, 
        all_fma_preds, 
        model_name="multitask_joint_fma",
        save_dir="Models/metrics"
    )

# 辅助函数：计算回归指标
def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy().flatten()
    y_pred = y_pred.cpu().detach().numpy().flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

# 辅助函数：保存回归指标
def calculate_regression_metrics(y_true, y_pred, model_name="model", save_dir="Models",file_name="fma_brs_log.txt"):
    """
    计算回归模型的核心指标并保存结果

    参数:
    y_true: 真实值数组 (numpy array 或 torch tensor)
    y_pred: 预测值数组 (numpy array 或 torch tensor)
    model_name: 模型名称（用于保存文件命名）
    save_dir: 结果保存目录
    """
    # 1. 统一转换为numpy数组（处理列表和张量情况）
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    elif isinstance(y_true, torch.Tensor):
        # 若为张量，先转移到CPU并分离计算图
        y_true = y_true.cpu().detach().numpy()

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    elif isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    # 展平数组（处理多维输出情况，如形状为(batch, 1)）
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")

    # 整理结果
    metrics = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epoch + 1,
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "r2_score": round(r2, 6)
    }

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存为文本文件（可读性好）
    txt_path = os.path.join(save_dir, file_name)
    with open(txt_path, "a", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("回归模型评估指标\n")
        f.write("=" * 30 + "\n")

    print(f"指标计算完成，结果已保存到 {save_dir} 目录")
    return metrics

def append_confusion_matrix_with_timestamp(metrics, loop_num, save_dir="Models", file_name="fma_brs_log.txt", is_percent=True):
    """
    追加混淆矩阵到txt文件，并添加当前时间戳

    参数：
        conf_matrix: 混淆矩阵（NumPy数组）
        file_path: 保存文件路径
        is_percent: 是否为百分比格式（True则保留两位小数）
    """
    # 获取当前时间戳（格式：年-月-日 时:分:秒）
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建metrics字符串（按固定格式拼接）
    metrics_str = (
        f"[时间: {timestamp}] "
        f"[循环: {loop_num if loop_num is not None else 'N/A'}] "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Macro-P: {metrics['macro']['precision']:.4f}, Macro-R: {metrics['macro']['recall']:.4f}, Macro-F1: {metrics['macro']['f1']:.4f} | "
        f"Micro-P: {metrics['micro']['precision']:.4f}, Micro-R: {metrics['micro']['recall']:.4f}, Micro-F1: {metrics['micro']['f1']:.4f}"
    )

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "a", encoding="utf-8") as f:
        # 写入时间戳和分隔线
        f.write(metrics_str + "\n")  # 每条记录占一行

def metrics_from_confusion(cm):
    """
    从混淆矩阵计算准确率、各类别精确率、召回率、F1，及宏平均/微平均指标

    参数：
        cm: 混淆矩阵（numpy数组，形状为 [n_classes, n_classes]）
    返回：
        metrics: 字典，包含各类指标
    """
    n_classes = cm.shape[0]
    total = cm.sum()

    # 初始化各类别指标列表
    precision = []
    recall = []
    f1 = []

    # 计算每个类别的精确率、召回率、F1
    for k in range(n_classes):
        tp = cm[k, k]  # 真正例
        fp = cm[:, k].sum() - tp  # 假正例
        fn = cm[k, :].sum() - tp  # 假负例

        # 避免除零错误（当分母为0时，指标设为0）
        p = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) != 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    # 计算准确率
    accuracy = cm.trace() / total if total != 0 else 0.0  # 对角线元素和为所有TP之和

    # 计算宏平均（各类别指标的算术平均）
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # 计算微平均（全局TP/FP/FN计算）
    total_tp = cm.trace()
    total_fp = cm.sum(axis=0).sum() - total_tp  # 所有预测为正的样本 - 总TP
    total_fn = cm.sum(axis=1).sum() - total_tp  # 所有真实为正的样本 - 总TP

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) != 0 else 0.0

    # 整理结果
    metrics = {
        "accuracy": accuracy,
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        }
    }
    return metrics
# 主函数
if __name__ == "__main__":
    train_loader, val_loader = create_multitask_dataloaders()
    
    # 加载检查点（如果需要）
    checkpoint_path = 'Models/checkpoints'
    modelname = os.path.join(checkpoint_path, 'xxx.pth')
    if os.path.exists(modelname):
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded model from {modelname}')
    
    # 训练循环
    for epoch in range(Epoch):
        train(epoch, loss_weights=[0.5, 0.5])  # 可调整任务权重
        val(epoch)
    
    # 最终评估
    #eval_model(fma_labels, fma_preds, brs_labels, brs_preds)
