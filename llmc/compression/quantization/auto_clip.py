import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from .module_utils import _LLMC_LINEAR_TYPES_, _TRANSFORMERS_LINEAR_TYPES_
from .utils import check_do_quant, check_w_only, get_aquantizer, get_wquantizer


class AutoClipper:
    """
    AutoClipper 类用于在量化过程中自动裁剪权重，以优化量化精度。

    该类实现了两种裁剪版本：
    - v1: 直接裁剪权重值到最优范围
    - v2: 计算并存储裁剪因子，用于后续的可学习裁剪

    主要功能：
    1. 通过网格搜索找到最优的权重裁剪范围
    2. 最小化量化前后的输出误差
    3. 支持对称和非对称裁剪
    4. 支持分布式训练
    """

    def __init__(
        self,
        w_only,  # 是否只量化权重（不量化激活）
        mix_bits_map,  # 混合比特映射
        quantizer_mix_bits,  # 量化器混合比特配置
        wquantizer,  # 权重量化器
        aquantizer,  # 激活量化器
        clip_version,  # 裁剪版本 ('v1' 或 'v2')
        clip_sym,  # 是否使用对称裁剪
        save_clip,  # 是否保存裁剪参数
        padding_mask,  # 填充掩码
    ):
        """
        初始化 AutoClipper

        Args:
            w_only: 是否只量化权重
            mix_bits_map: 混合比特映射配置
            quantizer_mix_bits: 量化器混合比特设置
            wquantizer: 权重量化器实例
            aquantizer: 激活量化器实例
            clip_version: 裁剪版本，'v1' 或 'v2'
            clip_sym: 是否使用对称裁剪
            save_clip: 是否保存裁剪参数到文件
            padding_mask: 用于处理填充token的掩码
        """
        self.mix_bits_map = mix_bits_map
        self.quantizer_mix_bits = quantizer_mix_bits
        self.wquantizer = wquantizer
        self.aquantizer = aquantizer
        self.clip_version = clip_version
        self.clip_sym = clip_sym
        self.save_clip = save_clip
        self.padding_mask = padding_mask
        self.weight_clips = {}  # 存储权重裁剪参数
        self.w_only = w_only
        self.logit = lambda x: torch.log(x / (1 - x))  # logit函数，用于计算裁剪因子

    @torch.no_grad()
    def run(self, block, block_idx, input_feat, n_sample_token):
        """
        对给定的block执行自动裁剪

        Args:
            block: 要处理的模型块
            block_idx: 块索引
            input_feat: 输入特征字典，包含每层的激活值
            n_sample_token: 用于裁剪优化的采样token数量
        """
        # 遍历block中的所有模块
        for n, m in block.named_modules():
            # 检查该层是否需要量化
            if not check_do_quant(
                block_idx, n, self.mix_bits_map, self.quantizer_mix_bits
            ):
                logger.info(
                    f"This layer {n} in {block_idx}-th block is set to float."
                    f"No need to clip this layer."
                )
                continue

            # 只处理线性层
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
                m = m.cuda()

                # 跳过注意力机制中的q、k层，因为它们通常不需要裁剪
                if any([_ in n for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                    if self.clip_version == "v2":
                        # 为v2版本注册空的缓冲区
                        m.register_buffer("buf_upbound_factor", None)
                        m.register_buffer("buf_lowbound_factor", None)
                    continue

                logger.info(f"clip layer: {n}")

                # 准备输入数据
                inputs = (
                    [torch.cat(input_feat[n])]
                    if len(input_feat[n]) != 1
                    else input_feat[n]
                )

                # 计算该层的最优裁剪范围
                max_val, min_val = self.auto_clip_layer(
                    block_idx, n, m.weight, inputs, n_sample_token=n_sample_token
                )

                # 分布式训练时同步裁剪参数
                dist.all_reduce(max_val, op=dist.ReduceOp.SUM)
                max_val /= int(os.environ["WORLD_SIZE"])

                dist.all_reduce(min_val, op=dist.ReduceOp.SUM)
                min_val /= int(os.environ["WORLD_SIZE"])

                # 应用裁剪
                self.apply_clip(block_idx, m, min_val, max_val, n)

    @torch.no_grad()
    def auto_clip_layer(
        self,
        block_idx,
        layer_name,
        w,  # 权重张量
        inputs,  # 输入特征列表
        n_grid=20,  # 网格搜索的步数
        max_shrink=0.5,  # 最大收缩比例
        n_sample_token=512,  # 采样token数量
        eps=0.0,  # 数值稳定性参数
    ):
        """
        为单个层自动计算最优裁剪范围

        通过网格搜索找到最优的权重裁剪范围，使得量化前后的输出误差最小。

        Args:
            block_idx: 块索引
            layer_name: 层名称
            w: 权重张量 (shape: [out_features, in_features])
            inputs: 输入特征列表
            n_grid: 网格搜索的步数
            max_shrink: 最大收缩比例
            n_sample_token: 用于计算的采样token数量
            eps: 数值稳定性参数

        Returns:
            tuple: (最优最大值, 最优最小值)
        """
        assert w.dim() == 2  # 确保权重是2D张量

        # 获取该层的权重量化器
        wquantizer = get_wquantizer(
            block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )

        # 确定分组大小
        if wquantizer.granularity == "per_group":
            group_size = wquantizer.group_size
        else:
            group_size = w.shape[1]

        # 重塑权重张量以适应分组量化
        try:
            w = w.reshape(w.shape[0], 1, -1, group_size)
        except RuntimeError:
            w = self.wquantizer.reshape_tensor(w)
            w = w.reshape(w.shape[0], 1, -1, group_size)

        # 设置批次大小以防止OOM
        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
        assert w.shape[0] % oc_batch_size == 0

        w_all = w
        best_max_val_all, best_min_val_all = [], []

        # 分批处理权重以节省内存
        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            # 计算原始的最大值和最小值
            if self.clip_sym:
                org_max_val = w.abs().amax(dim=-1, keepdim=True)
            else:
                org_max_val = w.amax(dim=-1, keepdim=True)

            org_min_val = w.amin(dim=-1, keepdim=True)

            # 初始化最优值
            best_max_val = org_max_val.clone()
            best_min_val = org_min_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9  # 初始化为很大的误差
            org_out_dict = {}  # 缓存原始输出

            # 网格搜索最优裁剪范围
            for i_s in range(int(max_shrink * n_grid)):
                if i_s == 0:
                    # 特殊处理第一步
                    if self.clip_version == "v2" and not check_w_only(
                        block_idx,
                        layer_name,
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.w_only,
                    ):
                        i_s += eps

                err_mean = 0

                # 对每个输入计算误差
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(w.device)
                    x = inputs[i]
                    x = x.view(-1, x.shape[-1])

                    # 应用填充掩码
                    if self.padding_mask and self.padding_mask[i].numel() == x.shape[0]:
                        mask_tmp = self.padding_mask[i].flatten()
                        x = x[mask_tmp.bool()]

                    # 重塑输入以匹配权重的形状
                    try:
                        x = x.reshape(1, x.shape[0], -1, group_size)
                    except RuntimeError:
                        x = self.wquantizer.reshape_tensor(x)
                        x = x.reshape(1, x.shape[0], -1, group_size)

                    # 采样token以减少计算量
                    if n_sample_token is None:
                        n_sample_token = min(x.shape[1], 512)
                    x = x[:, 0 :: x.shape[1] // n_sample_token]

                    # 计算或获取原始输出
                    if i in org_out_dict:
                        org_out = org_out_dict[i]
                    else:
                        org_out = (x * w).sum(dim=-1)
                        org_out_dict[i] = org_out

                    # 计算当前步的裁剪范围
                    max_val = org_max_val * (1 - i_s / n_grid)

                    if self.clip_sym:
                        min_val = -max_val
                    else:
                        min_val = org_min_val * (1 - i_s / n_grid)

                    # 执行伪量化
                    q_w = self.fake_quantize_weight(
                        w, min_val, max_val, org_min_val, org_max_val
                    )
                    q_x = self.fake_quantize_input(block_idx, x, layer_name)

                    # 计算量化后的输出
                    cur_out = (q_x * q_w).sum(dim=-1)

                    # 计算误差 (shape: co, 1, n_group, 1)
                    err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                    err_mean += err

                    del cur_out

                err_mean /= len(inputs)

                # 更新最优值
                cur_best_idx = err_mean < min_errs
                min_errs[cur_best_idx] = err_mean[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_min_val[cur_best_idx] = min_val[cur_best_idx]

            best_max_val_all.append(best_max_val)
            best_min_val_all.append(best_min_val)

        # 合并所有批次的结果
        best_max_val = torch.cat(best_max_val_all, dim=0)
        best_min_val = torch.cat(best_min_val_all, dim=0)

        # 清理内存
        del org_out
        del org_out_dict
        gc.collect()
        torch.cuda.empty_cache()

        return best_max_val.squeeze(1), best_min_val.squeeze(1)

    @torch.no_grad()
    def apply_clip(self, block_idx, layer, min_val, max_val, layer_name):
        """
        apply clippings to weights

        Args:
            block_idx: 块索引
            layer: 要裁剪的层
            min_val: 最小裁剪值
            max_val: 最大裁剪值
            layer_name: 层名称
        """
        if self.clip_version in ("v1", "v3", "v4"):
            # v1版本：直接裁剪权重值
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape

            # 重塑权重以匹配裁剪值的形状
            try:
                layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            except RuntimeError:
                layer.weight.data = self.wquantizer.reshape_tensor(layer.weight.data)
                layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)

            # 对称裁剪
            if self.clip_sym:
                min_val = -max_val

            # 执行裁剪
            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)

            # 恢复原始形状
            try:
                layer.weight.data = layer.weight.data.reshape(org_shape)
            except RuntimeError:
                layer.weight.data = self.wquantizer.restore_tensor(
                    layer.weight.data, org_shape
                )

        elif self.clip_version == "v2":
            # v2版本：计算并存储裁剪因子
            up_factor, low_factor = self.get_clip_factor(
                block_idx, layer, min_val, max_val, layer_name
            )

            # 注册裁剪因子为缓冲区
            layer.register_buffer("buf_upbound_factor", up_factor)
            layer.register_buffer("buf_lowbound_factor", low_factor)

            # 保存裁剪参数（如果需要）
            if self.save_clip:
                if block_idx not in self.weight_clips:
                    self.weight_clips[block_idx] = dict()
                n = f"{layer_name}.weight_quantizer."
                self.weight_clips[block_idx][n + "upbound_factor"] = up_factor.cpu()
                if low_factor is not None:
                    self.weight_clips[block_idx][n + "lowbound_factor"] = (
                        low_factor.cpu()
                    )
                else:
                    self.weight_clips[block_idx][n + "lowbound_factor"] = None
        else:
            raise Exception("Not support other clip version")

    def get_clip_factor(self, block_idx, layer, min_val, max_val, layer_name):
        """
        计算裁剪因子（用于v2版本）

        将裁剪值转换为logit空间的因子，用于可学习的裁剪

        Args:
            block_idx: 块索引
            layer: 目标层
            min_val: 最小裁剪值
            max_val: 最大裁剪值
            layer_name: 层名称

        Returns:
            tuple: (上界因子, 下界因子)
        """
        # 获取权重量化器
        wquantizer = get_wquantizer(
            block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )

        # 获取原始权重的最值范围
        org_min_val, org_max_val = wquantizer.get_minmax_range(
            wquantizer.reshape_tensor(layer.weight.data)
        )
        org_val_shape = org_max_val.shape

        if self.clip_sym:
            # 对称裁剪：只需要上界因子
            abs_max_val = torch.max(org_max_val.abs(), org_min_val.abs())
            abs_max_val = abs_max_val.clamp(min=1e-5)  # 防止除零
            abs_max_val = abs_max_val.reshape(*max_val.shape[:2], -1)

            # 计算logit空间的上界因子
            up_factor = self.logit((max_val / abs_max_val))
            up_factor = up_factor.reshape(org_val_shape)
            low_factor = None
        else:
            # 非对称裁剪：需要上界和下界因子
            org_max_val = org_max_val.reshape(*max_val.shape[:2], -1)
            up_factor = self.logit((max_val / org_max_val))
            up_factor = up_factor.reshape(org_val_shape)

            org_min_val = org_min_val.reshape(*min_val.shape[:2], -1)
            low_factor = self.logit((min_val / org_min_val))
            low_factor = low_factor.reshape(org_val_shape)

        return up_factor, low_factor

    def fake_quantize_weight(self, w, min_val, max_val, org_min_val, org_max_val):
        """
        执行权重的伪量化

        Args:
            w: 权重张量
            min_val: 裁剪最小值
            max_val: 裁剪最大值
            org_min_val: 原始最小值
            org_max_val: 原始最大值

        Returns:
            torch.Tensor: 伪量化后的权重
        """
        if self.clip_version == "v1":
            # v1版本：先裁剪再量化
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = self.wquantizer.fake_quant_weight_dynamic(cur_w)
        elif self.clip_version == "v2":
            # v2版本：使用裁剪因子进行可学习裁剪
            low_factor = self.logit((min_val / org_min_val))
            up_factor = self.logit((max_val / org_max_val))

            # 获取可学习的量化范围
            tensor_range = self.wquantizer.get_learnable_range(w, low_factor, up_factor)

            # 计算量化参数
            scales, zeros, qmax, qmin = self.wquantizer.get_qparams(
                tensor_range, w.device
            )
            args = {"scales": scales, "zeros": zeros, "qmax": qmax, "qmin": qmin}
            q_w = self.wquantizer.fake_quant_weight_static(w, args)
        else:
            raise Exception("Not support other clip version")
        return q_w

    def fake_quantize_input(self, block_idx, x, layer_name):
        """
        执行输入激活的伪量化

        Args:
            block_idx: 块索引
            x: 输入张量
            layer_name: 层名称

        Returns:
            torch.Tensor: 伪量化后的输入（如果不是仅权重量化）
        """
        # 检查是否只量化权重
        if not check_w_only(
            block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.w_only,
        ):
            # 同时量化激活
            q_x = get_aquantizer(
                block_idx,
                layer_name,
                self.mix_bits_map,
                self.quantizer_mix_bits,
                self.aquantizer,
            ).fake_quant_act_dynamic(x)
        else:
            # 仅权重量化，激活保持不变
            q_x = x
        return q_x


class AutoClipperV3(AutoClipper):
    """
    AutoClipperV2 类是 AutoClipper 的改进版本，使用矩阵乘法和MSE loss来计算最优裁剪范围。

    主要改进：
    1. 不需要分片处理权重，简化了内存管理
    2. 使用矩阵乘法直接计算层输出
    3. 参考 get_mse_range 的网格搜索逻辑来优化裁剪范围
    4. 使用MSE loss直接评估量化前后的输出差异
    """

    @torch.no_grad()
    def auto_clip_layer(
        self,
        block_idx,
        layer_name,
        w,  # 权重张量
        inputs,  # 输入特征列表
        n_grid=100,  # 网格搜索的步数，参考get_mse_range的默认值
        max_shrink=0.8,  # 最大收缩比例，参考get_mse_range的maxshrink
        n_sample_token=512,  # 采样token数量
        norm=2.4,  # MSE计算的范数，参考get_mse_range
    ):
        assert w.dim() == 2  # 确保权重是2D张量

        # 获取该层的权重量化器
        wquantizer = get_wquantizer(
            block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )

        # 重塑权重张量以适配量化器
        w_reshaped = wquantizer.reshape_tensor(w)

        # 获取初始的最大值和最小值范围
        min_val, max_val = wquantizer.get_minmax_range(w_reshaped)

        # 初始化最优值和最小误差
        best_min_val = min_val.clone()
        best_max_val = max_val.clone()
        best_errors = torch.full_like(max_val, float("inf"))

        # 预计算原始输出作为参考
        processed_inputs, original_outputs = (
            self._preprocess_inputs_and_compute_outputs(inputs, w, n_sample_token)
        )

        # 网格搜索最优裁剪范围（参考get_mse_range的搜索逻辑）
        for i in range(int(max_shrink * n_grid)):
            # 计算收缩比例
            p = 1 - i / n_grid

            # 计算当前步的裁剪范围
            if self.clip_sym:
                # 对称裁剪
                abs_max = torch.max(max_val.abs(), min_val.abs())
                cur_max_val = p * abs_max
                cur_min_val = -cur_max_val
            else:
                # 非对称裁剪
                cur_max_val = p * max_val
                cur_min_val = p * min_val

            # 计算当前裁剪范围下的总误差
            total_error = torch.zeros_like(best_errors)

            for j, (x, orig_out) in enumerate(zip(processed_inputs, original_outputs)):
                # 对权重执行伪量化
                q_w = self.fake_quantize_weight_with_clip(
                    w, w_reshaped, cur_min_val, cur_max_val, wquantizer
                )

                # 对输入执行伪量化（如果不是仅权重量化）
                q_x = self.fake_quantize_input(block_idx, x, layer_name)

                # 计算量化后的输出
                quantized_output = F.linear(q_x, q_w.to(q_x.dtype))

                # 计算MSE误差
                error = (quantized_output - orig_out).pow(norm)

                # 根据量化器的粒度聚合误差
                if wquantizer.granularity == "per_tensor":
                    error = error.mean().view(1, 1)
                elif wquantizer.granularity == "per_group":
                    # 按组聚合误差
                    group_size = wquantizer.group_size
                    error = error.mean(dim=0)  # [out_features]
                    # 重塑为组形状
                    num_groups = w.shape[1] // group_size
                    error = error.view(w.shape[0], num_groups).mean(dim=0, keepdim=True)
                else:
                    # per_channel: 按输出通道聚合
                    error = error.mean(dim=0, keepdim=True)  # [1, out_features]

                # 确保error形状与best_errors匹配
                if error.shape != best_errors.shape:
                    error = error.expand_as(best_errors)

                total_error += error

            # 平均所有输入的误差
            total_error /= len(processed_inputs)

            # 更新最优值
            improve_mask = total_error < best_errors
            if torch.any(improve_mask):
                best_errors[improve_mask] = total_error[improve_mask]
                best_max_val[improve_mask] = cur_max_val[improve_mask]
                best_min_val[improve_mask] = cur_min_val[improve_mask]

        # 清理内存
        del original_outputs, processed_inputs
        gc.collect()
        torch.cuda.empty_cache()

        return best_max_val, best_min_val

    def fake_quantize_weight_with_clip(
        self, w, w_reshaped, min_val, max_val, wquantizer
    ):
        """
        使用给定的裁剪范围对权重进行伪量化

        Args:
            w: 原始权重张量
            w_reshaped: 重塑后的权重张量
            min_val: 裁剪最小值
            max_val: 裁剪最大值
            wquantizer: 权重量化器

        Returns:
            torch.Tensor: 伪量化后的权重，恢复为原始形状
        """
        # 先对重塑后的权重进行裁剪
        clipped_w = torch.clamp(w_reshaped, min_val, max_val)

        # 计算裁剪范围的量化参数
        tensor_range = (min_val, max_val)
        scales, zeros, qmax, qmin = wquantizer.get_qparams(tensor_range, w.device)

        # 执行伪量化
        q_w = wquantizer.quant_dequant(clipped_w, scales, zeros, qmax, qmin)

        # 恢复到原始形状
        q_w = wquantizer.restore_tensor(q_w, w.shape)

        return q_w

    def _preprocess_inputs_and_compute_outputs(self, inputs, w, n_sample_token):
        """
        预处理输入数据并计算原始输出

        Args:
            inputs: 输入特征列表
            w: 权重张量
            device: 目标设备
            n_sample_token: 采样token数量

        Returns:
            tuple: (processed_inputs, original_outputs)
                - processed_inputs: 预处理后的输入列表
                - original_outputs: 原始输出列表
        """
        original_outputs = []
        processed_inputs = []

        for input_tensor in inputs:
            x = input_tensor.to(w.device)
            x = x.view(-1, x.shape[-1])  # 展平为2D

            # 应用填充掩码
            if self.padding_mask and len(self.padding_mask) > 0:
                if (
                    hasattr(self.padding_mask[0], "numel")
                    and self.padding_mask[0].numel() == x.shape[0]
                ):
                    mask_tmp = self.padding_mask[0].flatten()
                    x = x[mask_tmp.bool()]

            # 采样token以减少计算量
            if n_sample_token is not None and x.shape[0] > n_sample_token:
                indices = torch.randperm(x.shape[0])[:n_sample_token]
                x = x[indices]

            processed_inputs.append(x)

            original_output = F.linear(x, w.to(x.dtype))
            original_outputs.append(original_output)

        return processed_inputs, original_outputs


class AutoClipperV4(AutoClipper):
    """
    AutoClipperV4 类完全参考 get_mse_range 的实现，不使用输入激活。

    主要特点：
    1. 完全基于权重张量进行优化，不需要输入激活
    2. 直接参考 get_mse_range 的网格搜索和MSE计算逻辑
    3. 通过量化前后权重的MSE误差来找到最优裁剪范围
    4. 支持分批处理以节省内存
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_b_num = 1
        self.norm = 2.4

    @torch.no_grad()
    def auto_clip_layer(
        self,
        block_idx,
        layer_name,
        w,  # 权重张量
        inputs,  # 输入特征列表
        n_grid=20,  # 网格搜索的步数
        max_shrink=0.5,  # 最大收缩比例
        n_sample_token=512,  # 采样token数量
        eps=0.0,  # 数值稳定性参数
    ):
        """
        完全参考get_mse_range的实现来计算最优裁剪范围

        不使用输入激活，仅基于权重张量的量化MSE误差进行优化。
        这种方法更加简洁，不依赖于特定的输入数据。

        Args:
            block_idx: 块索引
            layer_name: 层名称
            w: 权重张量 (shape: [out_features, in_features])
            inputs: 输入特征列表（不使用）
            norm: MSE计算的范数

        Returns:
            tuple: (最优最大值, 最优最小值)
        """
        assert w.dim() == 2  # 确保权重是2D张量

        # 获取该层的权重量化器
        wquantizer = get_wquantizer(
            block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )

        # 重塑权重张量以适配量化器
        tensor = wquantizer.reshape_tensor(w).float()

        # 检查分批处理的有效性
        assert self.mse_b_num >= 1 and tensor.shape[0] % self.mse_b_num == 0, (
            f"Batch number {self.mse_b_num} must be divisible by tensor.shape[0] {tensor.shape[0]}"
        )

        # 计算批次大小
        bs = tensor.shape[0] // self.mse_b_num

        # 获取初始的最大值和最小值范围
        min_val, max_val = wquantizer.get_minmax_range(tensor)

        device = tensor.device

        # 分批处理权重（参考get_mse_range的分批逻辑）
        for b_num in range(self.mse_b_num):
            # 提取当前批次的张量和范围
            _tensor = tensor[b_num * bs : (b_num + 1) * bs, :]
            _min_val = min_val[b_num * bs : (b_num + 1) * bs, :]
            _max_val = max_val[b_num * bs : (b_num + 1) * bs, :]

            # 初始化最佳结果跟踪
            best = torch.full([_tensor.shape[0]], float("inf"), device=device)
            best_min_val = _min_val.clone()
            best_max_val = _max_val.clone()

            # 网格搜索最优裁剪范围（完全参考get_mse_range）
            for i in range(int(max_shrink * n_grid)):
                # 计算收缩比例
                p = 1 - i / n_grid

                # 计算当前步的裁剪范围
                if self.clip_sym:
                    # 对称裁剪：只考虑最大绝对值
                    abs_max = torch.max(_max_val.abs(), _min_val.abs())
                    xmax = p * abs_max
                    xmin = -xmax
                else:
                    # 非对称裁剪
                    xmin = p * _min_val
                    xmax = p * _max_val

                # 计算量化参数并执行伪量化
                scales, zeros, qmax, qmin = wquantizer.get_qparams((xmin, xmax), device)
                q_tensor = wquantizer.quant_dequant(_tensor, scales, zeros, qmax, qmin)

                # 计算MSE误差（完全参考get_mse_range的计算方式）
                q_tensor -= _tensor  # 计算差异
                q_tensor.abs_()  # 取绝对值
                q_tensor.pow_(self.norm)  # 应用范数
                err = torch.sum(q_tensor, 1)  # 按行求和得到每行的误差

                # 更新最佳结果
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    best_min_val[tmp] = xmin[tmp]
                    best_max_val[tmp] = xmax[tmp]

            # 更新全局的最值结果
            min_val[b_num * bs : (b_num + 1) * bs, :] = best_min_val
            max_val[b_num * bs : (b_num + 1) * bs, :] = best_max_val

        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

        return max_val, min_val