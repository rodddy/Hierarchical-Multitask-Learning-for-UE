import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention1D(nn.Module):
    """1维通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=3):
        super(ChannelAttention1D, self).__init__()
        # 1维自适应池化，压缩空间维度
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 共享的MLP结构，用于捕获通道间的依赖关系
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x的形状: (batch_size, channels, length)
        avg_out = self.fc(self.avg_pool(x))  # 平均池化路径
        max_out = self.fc(self.max_pool(x))  # 最大池化路径
        out = avg_out + max_out  # 特征融合
        return self.sigmoid(out)  # 输出通道注意力权重

class SpatialAttention1D(nn.Module):
    """1维空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3, 5, or 7 for 1D attention'
        padding = kernel_size // 2  # 保持长度不变

        # 1维卷积用于提取空间注意力特征
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x的形状: (batch_size, channels, length)
        # 在通道维度上计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大特征

        # 拼接两个特征图
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 形状变为 (batch_size, 2, length)

        # 通过卷积得到空间注意力图
        out = self.conv(x_cat)
        return self.sigmoid(out)  # 输出空间注意力权重

class CBAM1D(nn.Module):
    """1维CBAM模块，结合通道注意力和空间注意力"""
    def __init__(self, in_channels, reduction_ratio=3, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.channel_attention = ChannelAttention1D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention1D(kernel_size)

    def forward(self, x):
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        return x
# 示例：将1维CBAM模块与卷积块结合使用
class Conv1DBlockWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 reduction_ratio=3, attention_kernel=7):
        super(Conv1DBlockWithCBAM, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM1D(out_channels, reduction_ratio, attention_kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)  # 应用1维CBAM注意力
        return x

class TemporalAttention1D(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention1D, self).__init__()

        # 第一个卷积层: Out channels=3, Kernel size=1, Stride=1, Padding=0
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 第二个卷积层: Out channels=1, Kernel size=3, Stride=1, Padding=1
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 批归一化和ReLU
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x的形状: (batch_size, in_channels, time_steps)

        # 第一个卷积操作
        attn = self.conv1(x)  # 输出形状: (batch_size, 3, time_steps)

        # 计算均值 (dim=1)
        mean = torch.mean(attn, dim=1, keepdim=True)  # 输出形状: (batch_size, 1, time_steps)

        # 第二个卷积操作和Sigmoid激活
        attn_weights = self.conv2(attn)  # 输出形状: (batch_size, 1, time_steps)
        attn_weights = self.sigmoid(attn_weights)  # 输出形状: (batch_size, 1, time_steps)

        # 元素级乘法 (输入x与注意力权重)
        out = x * attn_weights  # 输出形状: (batch_size, in_channels, time_steps)

        # 批归一化和ReLU激活
        out = self.batch_norm(out)
        out = self.relu(out)

        return out
class CNNLSTMSTFusion(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, num_layers, num_classes=10):
        super(CNNLSTMSTFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_units_lstm = hidden_dim * 9

        # 2层卷积层
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=kernel_size, stride=2, padding=4)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=kernel_size//2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        self.cbam = CBAM1D(hidden_dim)
        self.temporal_attention = TemporalAttention1D(hidden_dim)
        self.flatten = nn.Flatten(start_dim=2)

        self.lstm1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        # x的形状: (batch_size, in_channels, time_steps)

        # 2层卷积层
        #print('raw : ' + str(x.shape))
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        #print('cnn1 : ' + str(x.shape))
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        #print('cnn2 : ' + str(x.shape))

        x = self.cbam(x) + self.temporal_attention(x)
        #print('cbam+ta : ' + str(x.shape))

        x = x.permute(0, 2, 1)
        #print('1.' + str(x.shape))
        x = self.flatten(x)
        #print('2.' + str(x.shape))
        lstm_out, _ = self.lstm1(x)
        #print('lstm1.' + str(lstm_out.shape))
        lstm_out, _ = self.lstm2(lstm_out)
        #print('lstm2.' + str(lstm_out.shape))
        # 取 LSTM 的最后一层输出
        lstm_out = lstm_out[:, -1, :]
        #print('lstm last.' + str(lstm_out.shape))
        # 全连接层
        out = self.fc(lstm_out)
        return out