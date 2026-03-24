import torch
from torch import nn
import torch.nn.functional as F


# --- 对应图中：HGC (Hypergraph Convolution) 和 Fusion ---
class HGNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super(HGNN, self).__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x, G):
        # [对应图中：从左侧连向右侧 Fusion 的橙色虚线]
        residual = x

        # [对应图中：HGC 模块]
        x = self.conv(x)  # 线性变换
        x = G.matmul(x)  # 超图卷积运算 (特征聚合)

        # [对应图中：Fusion]
        # 将卷积结果与残差相加
        x = F.relu(self.bn(x.permute(0, 2, 1).contiguous())).permute(0, 2, 1).contiguous() + residual
        return x


# --- 对应图中：Hypergraph Construction 和 Hypergraph ---
class HGNN_layer(nn.Module):
    """
        Written by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, node=None, K_neigs=None, kernel_size=5, stride=2):
        super(HGNN_layer, self).__init__()
        self.HGNN = HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs

        # 这里调用了 local_kernel，所以下面必须定义它
        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        B, N, C = x.shape
        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(x, k=self.K_neigs[0])
        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)
        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.
        Dv = Dv * alpha
        max_k = int(Dv.max())
        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(x, k=max_k - 1)
        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(x.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(x.device)
        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()
        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)
        local_H = self.local_H.repeat(B, 1, 1).to(new_H.device)

        _H = torch.cat([new_H, local_H], dim=2)
        _G = self._generate_G_from_H_b(_H)

        x = self.HGNN(x, _G)

        return x

    # ================= 缺失的辅助函数 =================

    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape

        # the weight of the hyperedge
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)
        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedge
        DE = torch.sum(H, dim=1)

        invDE = torch.diag_embed((torch.pow(DE, -1)))
        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)

        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 @ H @ W @ invDE @ HT @ DV2
            return G

    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)

    @torch.no_grad()
    def batched_knn(self, x, k=1):
        ori_dists = self.pairwise_distance(x)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)
        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)
        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]
        incidence_matrix[batch_indices, pixel_indices, inds] = weights
        return incidence_matrix.permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            # Chai's weight function
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        # 这就是报错说找不到的那个函数
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]
        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()
        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()
        H_local = torch.zeros((size * size, edge))
        H_local[inp_unf, matrix] = 1.
        return H_local

# --- 对应图中：最外层的蓝色虚线框 (AHGNN Wrapper) ---
class HyperNet(nn.Module):
    def __init__(self, channel, node=28, kernel_size=3, stride=1, K_neigs=None):
        super(HyperNet, self).__init__()
        self.HGNN_layer = HGNN_layer(channel, node=node, kernel_size=kernel_size, stride=stride, K_neigs=K_neigs)

    def forward(self, x):
        b, c, w, h = x.shape

        # [对应图中：左侧的 Reshape]
        # (Batch, Channel, Width, Height) -> (Batch, Nodes, Channel)
        # 注意代码里有个 permute(0,2,1)，将 Channel 移到了最后，变成了 (B, N, C)
        x = x.view(b, c, -1).permute(0, 2, 1).contiguous()

        # 进入核心层
        x = self.HGNN_layer(x)

        # [对应图中：右侧的 Reshape]
        # (Batch, Nodes, Channel) -> (Batch, Channel, Width, Height)
        x = x.permute(0, 2, 1).contiguous().view(b, c, w, h)

        return x

if __name__ == '__main__':
    # 定义输入参数
    batch_size = 1
    channels = 512  # 模拟中间层特征通道数
    h, w = 65, 65  # 特征图尺寸 (必须与初始化时的 node 参数一致)

    # 1. 实例化模块
    # 注意: K_neigs 必须是列表格式，例如 [3]
    block = HyperNet(channel=channels, node=h, kernel_size=3, stride=1, K_neigs=[3])

    # 2. 创建模拟输入 (Batch, Channel, Height, Width)
    # AHGNN 内部根据特征自动计算距离构建图，所以不需要 edge_input
    input_tensor = torch.randn(batch_size, channels, h, w)

    # 3. 前向传播
    output_tensor = block(input_tensor)

    # 4. 打印结果
    print("Input size: ", input_tensor.size())  # 预期: torch.Size([2, 512, 28, 28])
    print("Output size:", output_tensor.size())  # 预期: torch.Size([2, 512, 28, 28])