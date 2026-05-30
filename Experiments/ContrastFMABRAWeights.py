import os.path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --------------------------
# 1. 配置参数（根据实际模型修改）
# --------------------------
base_dir = 'E:/python_works/ClinicalExperimentData/kmd_data/experiment/Models/CNNLSTMPRE/'

num_experts = 15  # 专家数量（k=15）
model_a_path = os.path.join(base_dir, 'model_multitask_FMA_epoch_80.pth')  # 模型A的.pth文件路径
model_b_path = os.path.join(base_dir, 'model_multitask_Brunnstrom_epoch_80.pth')  # 模型B的.pth文件路径
# 门控输出层参数名（需与模型state_dict中的实际名称一致）
gate_weight_key = "gate.fc.2.weight"  # 权重参数
gate_bias_key = "gate.fc.2.bias"  # 偏置参数


# --------------------------
# 2. 加载模型参数并提取门控网络特征
# --------------------------
def load_gate_params(model_path, weight_key, bias_key):
    """从.pth文件中提取门控网络的权重和偏置参数"""
    # 加载完整checkpoint（包含model_state_dict等）
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    print("模型参数列表（找到门控输出层参数名）：")
    print(list(state_dict.keys()))

    # 提取门控输出层的权重和偏置（处理嵌套结构）
    def get_nested_key(d, key):
        """获取嵌套字典中的参数（如"model_state_dict.gate.fc2.weight"）"""
        keys = key.split(".")
        print(keys)
        for k in keys:
            d = d[k]
        return d

    try:
        # 权重参数：shape=[num_experts, hidden_dim]
        # 提取输出层权重（shape通常为 [15, hidden_dim]，15对应15个专家）
        gate_weights = state_dict[weight_key].numpy()
        # 偏置参数：shape=[num_experts]
        gate_biases = state_dict[bias_key].numpy()
        return gate_weights, gate_biases
    except KeyError as e:
        raise ValueError(f"参数名不存在，请检查：{e}。可能需要修改gate_weight_key或gate_bias_key。")


# 加载模型A和B的门控参数
gate_weights_a, gate_biases_a = load_gate_params(model_a_path, gate_weight_key, gate_bias_key)
gate_weights_b, gate_biases_b = load_gate_params(model_b_path, gate_weight_key, gate_bias_key)

# 验证参数形状是否符合预期（[num_experts, ...]）
assert gate_weights_a.shape[0] == num_experts, f"模型A专家数量不符，预期{num_experts}，实际{gate_weights_a.shape[0]}"
assert gate_weights_b.shape[0] == num_experts, f"模型B专家数量不符，预期{num_experts}，实际{gate_weights_b.shape[0]}"


# --------------------------
# 3. 计算基于参数的专家重要性指标
# --------------------------
def calc_param_based_metrics(gate_weights, gate_biases):
    """仅通过门控参数计算专家重要性指标（无测试数据）"""
    # 指标1：权重L2范数（反映参数对输出的影响强度）
    weight_l2 = np.linalg.norm(gate_weights, axis=1)  # 每行（专家）的L2范数
    norm_l2 = weight_l2 / np.sum(weight_l2)  # 归一化（模型内相对重要性）

    # 指标2：权重绝对值均值（反映参数整体大小）
    weight_abs_mean = np.mean(np.abs(gate_weights), axis=1)
    norm_abs_mean = weight_abs_mean / np.sum(weight_abs_mean)

    # 指标3：偏置值（反映专家的基础激活倾向，正值更易被选中）
    # 偏置可能为负，需归一化到0~1范围
    bias_norm = (gate_biases - np.min(gate_biases)) / (np.max(gate_biases) - np.min(gate_biases) + 1e-8)

    # 指标4：权重稀疏度（非零参数占比，越高说明专家功能越复杂）
    weight_sparsity = np.mean(gate_weights != 0, axis=1)  # 假设0为稀疏值（可替换为接近0的阈值）

    return pd.DataFrame({
        "专家ID": range(num_experts),
        "归一化L2范数": norm_l2,
        "归一化绝对值均值": norm_abs_mean,
        "归一化偏置值": bias_norm,
        "权重稀疏度": weight_sparsity
    })


# 计算模型A和B的指标
df_a = calc_param_based_metrics(gate_weights_a, gate_biases_a)
df_a["Model "] = "A"

df_b = calc_param_based_metrics(gate_weights_b, gate_biases_b)
df_b["Model "] = "B"

# 合并两个模型的数据
combined_df = pd.concat([df_a, df_b], ignore_index=True)

# --------------------------
# 4. 跨模型对比可视化
# --------------------------
# 设置中文字体
plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["legend.fontsize"] = 16  # 字体大小
# 4.1 归一化L2范数对比（核心指标，反映参数影响强度）
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = np.arange(num_experts)

plt.bar(x - bar_width / 2, df_a["归一化L2范数"], width=bar_width, label="FMA Task", alpha=0.8)
plt.bar(x + bar_width / 2, df_b["归一化L2范数"], width=bar_width, label="BRS Task", alpha=0.8)


#plt.xlabel("Expert ID", fontsize=12)
plt.ylabel("L2 Norm (Normalized)", fontsize=18)
plt.title("Comparison of Expert Importance Between Two Models", fontsize=20)
plt.xticks(x, [f"Expert {i+1}" for i in range(num_experts)], rotation=45, fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

# 4.2 归一化偏置值对比（反映基础激活倾向）
plt.figure(figsize=(12, 6))
plt.bar(x - bar_width / 2, df_a["归一化偏置值"], width=bar_width, label="模型A", alpha=0.8)
plt.bar(x + bar_width / 2, df_b["归一化偏置值"], width=bar_width, label="模型B", alpha=0.8)

plt.xlabel("专家ID", fontsize=12)
plt.ylabel("归一化偏置值（越高越易被激活）", fontsize=12)
plt.title("两模型专家偏置值对比", fontsize=14)
plt.xticks(x, [f"专家{i}" for i in range(num_experts)], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 4.3 指标相关性散点图（L2范数 vs 绝对值均值）
plt.figure(figsize=(8, 6))
# 模型A的相关性
plt.scatter(df_a["归一化L2范数"], df_a["归一化绝对值均值"],
            c="blue", label="模型A", alpha=0.7, s=60)
# 模型B的相关性
plt.scatter(df_b["归一化L2范数"], df_b["归一化绝对值均值"],
            c="orange", label="模型B", alpha=0.7, s=60)

plt.xlabel("归一化L2范数", fontsize=12)
plt.ylabel("归一化绝对值均值", fontsize=12)
plt.title("专家参数指标相关性", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 4.4 权重稀疏度分布对比（箱线图）
plt.figure(figsize=(8, 6))
box_data = [
    combined_df[combined_df["Model "] == "A"]["权重稀疏度"],
    combined_df[combined_df["Model "] == "B"]["权重稀疏度"]
]
plt.boxplot(box_data, labels=["模型A", "模型B"], patch_artist=True)

plt.ylabel("权重稀疏度（越高功能越复杂）", fontsize=12)
plt.title("两模型专家权重稀疏度分布", fontsize=14)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------
# 5. 输出对比表格
# --------------------------
print("=== 两模型专家参数重要性对比表 ===")
comparison_table = combined_df.pivot(
    index="专家ID",
    columns="Model ",
    values=["归一化L2范数", "归一化偏置值", "权重稀疏度"]
).round(3)
print(comparison_table)