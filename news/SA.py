import torch
import torch.nn as nn
from news.lka import LKA

""" 
    论文地址：https://arxiv.org/abs/2501.03775
    论文题目：Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection (AAAI 2026) 
    中文题目：Strip R-CNN：用于遥感目标检测的大条带卷积 (AAAI 2026) 
    讲解视频：https://www.bilibili.com/video/BV1sHS5BuEae/
        条带卷积模块（Strip Convolution Module,SCM）
        实际意义：①传统方形卷积核的局限性：在提取特征时，传统检测器（方形卷积）容易包含无关信息，无法有效捕捉各向异性上下文，尤其是细长物体（一个维度上特征多，另一个维度上特征稀疏）。
                ②物体纵横比变异性大：当目标物体的纵横比差异显著，现有大核卷积计算开销大。
                ③定位能力的不足：全连接层或小核卷积的空间相关性有限，对高纵横比物体的边界回归和角度估计敏感度低，简单来说就是小角度误差会导致大偏差。
        实现方式：一个水平方向的条带卷积，一个垂直方向的条带卷积。
"""

class DirectionalStripeAttention(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()

        # 基础局部空间特征（5×5深度卷积）
        self.local_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 纵向条纹卷积：1×k2
        self.stripe_vertical = nn.Conv2d(
            dim, dim,
            kernel_size=(k1, k2),
            padding=(k1 // 2, k2 // 2),
            groups=dim
        )

        # 横向条纹卷积：k2×1
        self.stripe_horizontal = nn.Conv2d(
            dim, dim,
            kernel_size=(k2, k1),
            padding=(k2 // 2, k1 // 2),
            groups=dim
        )

        # 通道融合
        self.channel_mix = nn.Conv2d(dim, dim, 1)
        self.lka = LKA(512)

    def forward(self, x):
        attn = self.local_conv(x)
        attn = self.stripe_vertical(attn)
        attn = self.stripe_horizontal(attn)
        attn = self.channel_mix(attn)
        x = self.lka(x)
        attn = x + attn


        return x * attn

class StripeAttentionBlock(nn.Module):
    def __init__(self, d_model, k1=1, k2=19):
        super().__init__()

        self.proj_in = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()

        # 核心：条状方向注意力
        self.stripe_attn = DirectionalStripeAttention(d_model, k1, k2)

        self.proj_out = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.stripe_attn(x)
        x = self.proj_out(x)
        return x + shortcut

if __name__ == "__main__":
    input_tensor = torch.randn(1, 512, 65, 65)
    model = StripeAttentionBlock(d_model=512, k1=1, k2=19)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")