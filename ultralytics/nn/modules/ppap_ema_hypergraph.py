# Ultralytics YOLO 🚀, AGPL-3.0 license
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("PPAPEMA", "KDEHyperGraphFusion", "CLAGRGCUFuse")


def _norm_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    将 4D 特征图压成可视化热力图: [B,C,H,W] -> [B,1,H,W]
    使用 L2 范数聚合通道，再归一化到 [0,1]
    """
    if x.ndim != 4:
        raise ValueError(f"_norm_map expects 4D tensor, got {x.shape}")
    m = torch.norm(x, p=2, dim=1, keepdim=True)
    b = m.shape[0]
    flat = m.view(b, -1)
    mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (m - mn) / (mx - mn + eps)


class ConvBNAct(nn.Module):
    """Conv2d + BN + SiLU"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PPAP(nn.Module):
    """
    Peak-Preserving Average Pooling
    对每个通道取 top-k 响应再求均值，保留小目标/弱缺陷峰值响应
    """
    def __init__(self, topk_ratio: float = 0.05, topk_min: int = 8, topk_max: int = 64):
        super().__init__()
        assert 0 < topk_ratio <= 1.0
        self.topk_ratio = float(topk_ratio)
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        return: [B,C,1,1]
        """
        b, c, h, w = x.shape
        hw = h * w
        k = int(round(self.topk_ratio * hw))
        k = max(self.topk_min, min(self.topk_max, k))
        k = max(1, min(hw, k))

        flat = x.flatten(2)                         # [B,C,HW]
        topk_vals, _ = torch.topk(flat, k=k, dim=2, largest=True, sorted=False)
        out = topk_vals.mean(dim=2, keepdim=True)  # [B,C,1]
        return out.unsqueeze(-1)                    # [B,C,1,1]


class PPAPEMA(nn.Module):
    """
    符合原理图的 PPAP-EMA:
    1) 通道分组
    2) X/Y 轴向平均池化 + concat + 1x1 conv
    3) 双 Sigmoid 生成轴向门控
    4) 局部分支 3x3 conv
    5) 跨空间学习阶段使用 PPAP 替代 GAP
    6) 两路交叉 Softmax + Matmul
    7) 融合后 Sigmoid 生成最终空间权重图
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        topk_ratio: float = 0.05,
        topk_min: int = 8,
        topk_max: int = 64,
        gn_groups: int = 1,
    ):
        super().__init__()
        self.c = int(channels)

        g = int(groups)
        if g <= 0 or self.c % g != 0:
            g = 1
        self.G = g
        self.Cg = self.c // self.G

        gg = int(gn_groups)
        if gg <= 0 or self.Cg % gg != 0:
            gg = 1

        self.ppap = PPAP(topk_ratio=topk_ratio, topk_min=topk_min, topk_max=topk_max)

        # 局部分支
        self.local_conv = nn.Conv2d(self.Cg, self.Cg, kernel_size=3, stride=1, padding=1, bias=False)

        # 轴向聚合分支：concat(X_pool, Y_pool) 后走 1×1
        self.axial_fuse = nn.Conv2d(self.Cg, self.Cg, kernel_size=1, stride=1, padding=0, bias=True)

        self.gn = nn.GroupNorm(num_groups=gg, num_channels=self.Cg)

        # 仅在需要可视化时开启，避免训练时额外开销
        self.collect_visualization = False
        self.latest_vis: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _softmax_channel(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,1,1]
        return: [B,1,C]
        """
        b, c, _, _ = x.shape
        v = x.reshape(b, c)
        v = F.softmax(v, dim=1)
        return v.reshape(b, 1, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        """
        b, c, h, w = x.shape

        # 1) 通道分组
        xg = x.reshape(b, self.G, self.Cg, h, w).reshape(b * self.G, self.Cg, h, w)

        # 2) X/Y 轴向平均池化
        # 沿 H 聚合，保留 W
        x_pool_w = xg.mean(dim=2, keepdim=True)              # [BG, Cg, 1, W]
        # 沿 W 聚合，保留 H
        x_pool_h = xg.mean(dim=3, keepdim=True)              # [BG, Cg, H, 1]
        x_pool_h_t = x_pool_h.permute(0, 1, 3, 2)            # [BG, Cg, 1, H]

        # 3) concat + 1x1 conv
        x_hw = torch.cat([x_pool_w, x_pool_h_t], dim=3)      # [BG, Cg, 1, W+H]
        x_hw = self.axial_fuse(x_hw)

        a_w, a_h_t = torch.split(x_hw, [w, h], dim=3)
        a_h = a_h_t.permute(0, 1, 3, 2)                      # [BG, Cg, H, 1]

        # 4) 双 Sigmoid 轴向门控
        a_w = torch.sigmoid(a_w)
        a_h = torch.sigmoid(a_h)

        # 第一次 re-weight：符合你的原理图
        x_ax = xg * a_w * a_h

        # 5) 局部分支
        x_loc = self.local_conv(xg)

        # 6) 跨空间学习：左支做 GN 后再 PPAP
        x_ax_n = self.gn(x_ax)

        # 左路：Q_ax × K_loc
        q_ax = self._softmax_channel(self.ppap(x_ax_n))      # [BG,1,Cg]
        k_loc = x_loc.flatten(2)                             # [BG,Cg,HW]
        s_ax = torch.bmm(q_ax, k_loc).reshape(b * self.G, 1, h, w)

        # 右路：Q_loc × K_ax
        q_loc = self._softmax_channel(self.ppap(x_loc))      # [BG,1,Cg]
        k_ax = x_ax_n.flatten(2)                             # [BG,Cg,HW]
        s_loc = torch.bmm(q_loc, k_ax).reshape(b * self.G, 1, h, w)

        # 7) 融合空间权重图
        attn = torch.sigmoid(s_ax + s_loc)                   # [BG,1,H,W]

        # 最终 re-weight：按你的结构，对 x_ax 再做空间重标定
        y_group = x_ax * attn                                # [BG,Cg,H,W]
        y = y_group.reshape(b, self.G, self.Cg, h, w).reshape(b, c, h, w)

        if self.collect_visualization:
            with torch.no_grad():
                self.latest_vis = {
                    "input": _norm_map(x.detach()).cpu(),
                    "axial_reweight": _norm_map(
                        x_ax.reshape(b, self.G, self.Cg, h, w).reshape(b, c, h, w).detach()
                    ).cpu(),
                    "local_branch": _norm_map(
                        x_loc.reshape(b, self.G, self.Cg, h, w).reshape(b, c, h, w).detach()
                    ).cpu(),
                    "spatial_attn": _norm_map(
                        attn.reshape(b, self.G, 1, h, w).mean(dim=1).detach()
                    ).cpu(),
                    "output": _norm_map(y.detach()).cpu(),
                }

        return y


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    a: [B,T,C], b:[B,M,C]
    return: [B,T,M]
    """
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return torch.einsum("btc,bmc->btm", a_n, b_n)


def soft_kmeans_prototypes(
    tokens: torch.Tensor,
    num_nodes: int = 48,
    iters: int = 3,
    tau: float = 0.1,
    deterministic_init: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    轻量可微原型聚类
    tokens: [B,T,C]
    return:
        centers: [B,M,C]
        w:       [B,T,M]
    """
    b, t, c = tokens.shape
    m = max(1, min(int(num_nodes), t))

    if deterministic_init:
        idx = torch.linspace(0, t - 1, steps=m, device=tokens.device).long()
        centers = tokens[:, idx, :]
    else:
        ridx = torch.randperm(t, device=tokens.device)[:m]
        centers = tokens[:, ridx, :]

    centers = F.normalize(centers, dim=-1)
    w = None
    for _ in range(int(iters)):
        sim = cosine_sim(tokens, centers)                # [B,T,M]
        w = F.softmax(sim / float(tau), dim=-1)
        denom = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        num = torch.einsum("btm,btc->bmc", w, tokens)
        centers = F.normalize(num / denom, dim=-1).detach()
    return centers, w


class KDEHyperGraphConv(nn.Module):
    """
    严格贴合 Method 的 KDE 驱动超图传播：
    1) 计算节点间距离矩阵
    2) 用自适应带宽做 KDE，得到局部密度 rho
    3) 选取所有密度比当前顶点更高的节点作为邻域
    4) 邻域 + 自身 组成超边
    5) 用 H 做超图传播
    """

    def __init__(self, channels: int):
        super().__init__()
        self.c = int(channels)
        self.proj = nn.Linear(self.c, self.c, bias=False)
        self.act = nn.SiLU(inplace=True)

        self.collect_visualization = False
        self.latest_vis: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _build_incidence(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,M,C]
        return:
            H:    [B,M,M]   (vertex, hyperedge)
            dist: [B,M,M]
            rho:  [B,M]
        """
        b, m, _ = x.shape
        if m == 1:
            H = torch.ones((b, 1, 1), device=x.device, dtype=x.dtype)
            dist = torch.zeros((b, 1, 1), device=x.device, dtype=x.dtype)
            rho = torch.ones((b, 1), device=x.device, dtype=x.dtype)
            return H, dist, rho

        # 距离矩阵：符合 Method 中 d(z_i, z_j)
        dist = torch.cdist(x, x, p=2)  # [B,M,M]

        eye = torch.eye(m, device=x.device, dtype=torch.bool).unsqueeze(0)
        valid = ~eye
        dist_valid = dist.masked_select(valid).reshape(b, -1)

        # 自适应带宽：所有点对距离均值
        h = dist_valid.mean(dim=1, keepdim=True).clamp_min(1e-6)  # [B,1]
        h2 = (h ** 2).view(b, 1, 1)

        # 局部密度 rho_i = sum_j exp(-d_ij^2 / (2 h^2))
        rho = torch.exp(-(dist ** 2) / (2.0 * h2)).sum(dim=-1)    # [B,M]

        # 邻域筛选：选择密度更高的顶点
        # H[j, e_i] = 1, if rho_j > rho_i
        rho_member = rho.unsqueeze(2)  # [B,M,1]  -> j
        rho_edge = rho.unsqueeze(1)    # [B,1,M]  -> i
        higher = (rho_member > rho_edge)  # [B,M,M]

        H = higher.to(x.dtype)

        # 每条超边包含自身
        I = torch.eye(m, device=x.device, dtype=x.dtype).unsqueeze(0)
        H = torch.clamp(H + I, max=1.0)

        return H, dist, rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,M,C]
        """
        b, m, c = x.shape
        assert c == self.c

        H, dist, rho = self._build_incidence(x)

        # 经典超图传播
        # A = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
        De = H.sum(dim=1).clamp_min(1e-6)      # [B,E]
        Dv = H.sum(dim=2).clamp_min(1e-6)      # [B,V]
        De_inv = 1.0 / De
        Dv_inv_sqrt = 1.0 / torch.sqrt(Dv)

        H_De = H * De_inv.unsqueeze(1)
        A = torch.bmm(H_De, H.transpose(1, 2))
        A = A * Dv_inv_sqrt.unsqueeze(2) * Dv_inv_sqrt.unsqueeze(1)

        y = torch.bmm(A, x)                    # [B,M,C]
        y = self.proj(y)
        y = self.act(y)

        if self.collect_visualization:
            with torch.no_grad():
                # 这里只保留第一个 batch 的节点间结构图信息
                sim = 1.0 / (1.0 + dist.detach())
                self.latest_vis = {
                    "dist_matrix": dist[:1].detach().cpu(),
                    "sim_matrix": sim[:1].detach().cpu(),
                    "density": rho[:1].detach().cpu(),
                    "incidence": H[:1].detach().cpu(),
                }

        return y


class KDEHyperGraphCompute(nn.Module):
    """
    输入特征图 -> 原型聚类 -> KDE超图传播 -> 反投影 -> 残差增强
    """

    def __init__(self, channels: int, num_nodes: int = 48, cluster_iters: int = 3, tau: float = 0.1):
        super().__init__()
        self.c = int(channels)
        self.num_nodes = int(num_nodes)
        self.cluster_iters = int(cluster_iters)
        self.tau = float(tau)

        self.hyper = KDEHyperGraphConv(channels=self.c)
        self.out_proj = ConvBNAct(self.c, self.c, k=1, s=1)

        self.collect_visualization = False
        self.latest_vis: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        """
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B,T,C]

        # 原型节点
        _, w_assign = soft_kmeans_prototypes(
            tokens=tokens,
            num_nodes=self.num_nodes,
            iters=self.cluster_iters,
            tau=self.tau,
            deterministic_init=True,
        )

        denom = w_assign.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        nodes = torch.einsum("btm,btc->bmc", w_assign, tokens) / denom  # [B,M,C]

        # 超图传播
        nodes_out = self.hyper(nodes)

        # 反投影回 token
        tokens_out = torch.einsum("btm,bmc->btc", w_assign, nodes_out)
        feat_out = tokens_out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # 残差增强
        y = x + self.out_proj(feat_out)

        if self.collect_visualization:
            with torch.no_grad():
                self.latest_vis = {
                    "before_hg": _norm_map(x.detach()).cpu(),
                    "after_hg": _norm_map(y.detach()).cpu(),
                }

        return y


class KDEHyperGraphFusion(nn.Module):
    """
    完整版超图融合块：
    1) 收集多层特征
    2) 逐层 1x1 投影
    3) 对齐到目标尺度
    4) Concat + 1x1 fuse
    5) KDE 驱动超图传播
    6) 3x3 refine ×2
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        target_pos: int,
        num_nodes: int = 48,
        tau: float = 0.1,
    ):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.out_c = int(out_channels)
        self.target_pos = int(target_pos)

        branch_c = max(self.out_c // 2, 16)

        self.proj = nn.ModuleList([ConvBNAct(ci, branch_c, k=1, s=1) for ci in self.in_channels])
        self.fuse = ConvBNAct(branch_c * len(self.in_channels), self.out_c, k=1, s=1)

        self.hg = KDEHyperGraphCompute(
            channels=self.out_c,
            num_nodes=num_nodes,
            cluster_iters=3,
            tau=tau,
        )

        self.refine = nn.Sequential(
            ConvBNAct(self.out_c, self.out_c, k=3, s=1),
            ConvBNAct(self.out_c, self.out_c, k=3, s=1),
        )

        self.collect_visualization = False
        self.latest_vis: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _align(feat: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        h, w = feat.shape[-2:]
        th, tw = size
        if (h, w) == (th, tw):
            return feat
        if h > th or w > tw:
            return F.adaptive_avg_pool2d(feat, output_size=size)
        return F.interpolate(feat, size=size, mode="nearest")

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(feats, (list, tuple)) and len(feats) == len(self.in_channels)

        target_h, target_w = feats[self.target_pos].shape[-2:]
        xs = []
        for f, p in zip(feats, self.proj):
            y = p(f)
            y = self._align(y, (target_h, target_w))
            xs.append(y)

        x_cat = torch.cat(xs, dim=1)
        x_fuse = self.fuse(x_cat)
        x_hg = self.hg(x_fuse)
        out = self.refine(x_hg)

        if self.collect_visualization:
            with torch.no_grad():
                self.latest_vis = {
                    "before_hg": _norm_map(x_fuse.detach()).cpu(),
                    "after_hg": _norm_map(out.detach()).cpu(),
                }

        return out


class CLAG(nn.Module):
    """Cross-Layer Attention Guidance"""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNAct(channels, channels, k=1, s=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        guide = self.gate(shallow)
        return deep * guide, guide


class RGCU(nn.Module):
    """
    工程稳定版 RGCU：
    用门控重建替代简单插值后的直接拼接
    """

    def __init__(self, channels: int):
        super().__init__()
        self.reconstruct = nn.Sequential(
            ConvBNAct(channels, channels, k=3, s=1),
            ConvBNAct(channels, channels, k=3, s=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        rec = self.reconstruct(x)
        g = self.gate(rec + guide)
        return x + rec * g


class CLAGRGCUFuse(nn.Module):
    """
    Top-down 融合单元：
    deep -> upsample -> CLAG 引导 -> RGCU 重建 -> concat shallow -> 1x1 fuse
    """

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        c_deep, c_shallow = map(int, in_channels)
        self.out_c = int(out_channels)

        self.deep_proj = ConvBNAct(c_deep, self.out_c, k=1, s=1)
        self.shallow_proj = ConvBNAct(c_shallow, self.out_c, k=1, s=1)

        self.clag = CLAG(self.out_c)
        self.rgcu = RGCU(self.out_c)

        self.fuse = ConvBNAct(self.out_c * 2, self.out_c, k=1, s=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(feats, (list, tuple)) and len(feats) == 2
        deep, shallow = feats

        deep = self.deep_proj(deep)
        shallow = self.shallow_proj(shallow)

        if deep.shape[-2:] != shallow.shape[-2:]:
            deep = F.interpolate(deep, size=shallow.shape[-2:], mode="nearest")

        deep_guided, guide = self.clag(shallow, deep)
        deep_refined = self.rgcu(deep_guided, guide)

        out = self.fuse(torch.cat([deep_refined, shallow], dim=1))
        return out