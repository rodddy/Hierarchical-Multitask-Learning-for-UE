import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.sandbox.distributions.genpareto import meanexcess
from torch.jit import fuser

import Models.CNNLSTMSTFusion as CNNLSTMFusion

class Expert(nn.Module):
    """MoE中的专家网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class GatingNetwork(nn.Module):
    """门控网络"""

    def __init__(self, input_dim, num_experts, hidden_dim=64):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class MoEMHADeepConvLSTMConcate(nn.Module):
    """直接复用CNNLSTMSTFusion作为特征提取器的MoE模型"""

    def __init__(self,
                 input_channels=12,
                 hidden_dim=64,
                 kernel_size=5,
                 num_layers=2,
                 num_experts=15,
                 expert_hidden_dim=128,
                 weight_paths=None,
                 output_dim=10):
        super(MoEMHADeepConvLSTMConcate, self).__init__()

        self.num_experts = num_experts

        # 1. 初始化原有模型作为特征提取器
        self.experts = nn.ModuleList([
            CNNLSTMFusion.CNNLSTMSTFusion(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                num_classes=output_dim
            ) for _ in range(self.num_experts)
        ])

        self.load_pretrained_cnn(weight_paths)

        # 3. 计算特征维度（通过dummy input）
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 128)  # 假设输入形状
            # 获取CNN+注意力模块的输出特征
            cnn_feat = self._get_cnn_features(self.experts[0], dummy_input)
            self.feature_dim = cnn_feat.numel()  # 单个专家的特征维度
            self.gate_input_dim = self.feature_dim * self.num_experts  # 门控输入维度

        self.gate = GatingNetwork(self.gate_input_dim, num_experts)
        #print('self.gate_input_dim:', self.gate_input_dim)
        print('self.feature_dim:', self.feature_dim)
        # 3. 回归任务的MLP输出层（关键修改）
        self.mlp = nn.Sequential(
            nn.Linear(self.gate_input_dim, self.feature_dim//4),
            nn.BatchNorm1d(self.feature_dim//4),
            nn.ReLU(),
            nn.Dropout(0.75),
            # 回归任务输出层：无激活函数（直接预测连续值）
            nn.Linear(self.feature_dim//4, output_dim)
        )
        #self.fc= nn.Linear(self.feature_dim,output_dim)

    def _get_cnn_features(self, expert, x):
        """从CNNLSTMSTFusion中提取CNN+注意力模块的特征（移除LSTM层）"""
        # 复用原始模型的CNN层
        x = expert.conv1(x)
        x = expert.batch_norm1(x)
        x = expert.relu1(x)
        x = expert.pool1(x)
        x = expert.dropout1(x)

        x = expert.conv2(x)
        x = expert.batch_norm2(x)
        x = expert.relu2(x)
        x = expert.pool2(x)
        x = expert.dropout2(x)

        # 复用注意力模块
        x = expert.cbam(x) + expert.temporal_attention(x)
        x = expert.flatten(x)
        #print('x raw.' + str(x.shape))
        x = x.permute(0, 2, 1)
        #print('1.' + str(x.shape))
        x = expert.flatten(x)
        lstm_out, _ = expert.lstm1(x)
        #print('lstm1.' + str(lstmout.shape))
        lstm_out, _ = expert.lstm2(lstm_out)
        #print('lstm2.' + str(x.shape))
        lstm_out = lstm_out[:, -1, :]
        return lstm_out  # 输出CNN+注意力特征

    def get_ensemble(self, method='mean', expert_features=None):
        """获取集成特征"""
        if not expert_features:
            raise ValueError("No expert features added")
        if method == 'mean':
            return torch.mean(expert_features, dim=1)
        elif method == 'max':
            return torch.max(expert_features, dim=1)
        else:
            raise ValueError("Unsupported method")

    def forward(self, modalities):
        """
        输入: modalities - 包含15个模态的列表，每个形状为(batch_size, input_channels, time_steps)
        输出: 分类预测结果
        """
        modalities = modalities.permute(1, 0, 2, 3)
        assert len(modalities) == self.num_experts, f"需输入15个模态，实际输入{len(modalities)}个"
        # 1. 每个专家提取CNN+注意力特征（不经过LSTM）
        expert_features = []
        for i in range(self.num_experts):
            # 获取第i个模态的CNN特征
            feat = self._get_cnn_features(self.experts[i], modalities[i])
            # 展平特征用于后续融合
            expert_features.append(feat.flatten(1))  # 形状: (batch_size, feature_dim)
        #print(len(expert_features)+ self.feature_dim)

        # 2. 门控网络计算权重
        all_features = torch.cat(expert_features, dim=1)  # 拼接所有特征
        gate_weights = self.gate(all_features)  # 形状: (batch_size, num_experts)
        gate_weights = gate_weights.unsqueeze(2)  # 形状: (batch_size, num_experts, 1)
        #print(all_features.shape)
        # 3. 加权融合特征
        expert_features = torch.stack(expert_features, dim=1)  # 形状: (batch_size, num_experts, feature_dim)
        #fused_feature = torch.sum(gate_weights * expert_features, dim=1)  # 形状: (batch_size, feature_dim)

        #fused_feature = self.get_ensemble(method='mean', expert_features=expert_features)
        # 4. 分类预测
        # 1. concatenate
        #fused_feature = torch.cat(expert_features, dim=1)  # 形状: (batch_size, num_experts, feature_dim)
        # 2. mean
        #fused_feature = torch.mean(expert_features, dim=1)  # 形状: (batch_size, feature_dim)
        # 3. max
        #fused_feature, expert_indices = torch.max(expert_features, dim=1)  # 形状: (batch_size, feature_dim)
        #print(fused_feature.shape)

        out = self.mlp(all_features)
        #out = self.fc(fused_feature)
        return out

    def load_pretrained_cnn(self, cnn_weight_paths):
        """加载预训练的CNN权重（每个模态对应一个权重文件）"""

        assert len(cnn_weight_paths) == self.num_experts, \
            f"需要提供{self.num_experts}个模态的CNN权重文件，实际提供{len(cnn_weight_paths)}个"

        for i, path in enumerate(cnn_weight_paths):
            checkpoint = torch.load(path)
            model_state_dict = checkpoint['model_state_dict']  # 取出模型参数
            # 过滤掉fc层参数（因为类别数不匹配）
            filtered_state_dict = {
                k: v for k, v in model_state_dict.items()
                if not k.startswith('fc.')  # 排除fc层的所有参数
            }

            # 加载过滤后的参数（strict=False允许部分加载）
            self.experts[i].load_state_dict(filtered_state_dict, strict=False)
            # 冻结已加载的CNN层
            for param in self.experts[i].parameters():
                param.requires_grad = False