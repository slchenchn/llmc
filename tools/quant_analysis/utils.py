from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict
from loguru import logger


def calculate_kurtosis_channel(signal, eps=1e-8):
    """Calculates the kurtosis of a given signal along channels.

    Args:
        signal (torch.Tensor): Input signal of any shape. Calculation will be done
                              along the last dimension.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        float: The average kurtosis value across all channels.
    """
    # 确保使用浮点计算
    signal = signal.float()

    # 处理不同的输入形状
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)

    # 计算每个通道的统计量
    # 沿最后一个维度计算均值和方差
    mean = torch.mean(signal, dim=-1, keepdim=True)
    # 计算中心化的平方差（二阶矩）
    var = torch.var(signal, dim=-1, keepdim=True, unbiased=False)

    # 设置方差的下限，防止除零
    var = torch.clamp(var, min=eps)

    # 计算标准化后的信号
    centered = signal - mean

    # 计算四阶矩
    fourth_power_sum = torch.sum(centered**4, dim=-1)

    # 样本数量
    n = signal.shape[-1]

    # 峰度计算: (1/n) * sum((x-μ)^4) / (var^2)
    kurtosis = fourth_power_sum / (n * var.squeeze(-1) ** 2)

    # 返回所有通道平均值
    return torch.mean(kurtosis).item()


def calculate_kurtosis(signal, eps=1e-8):
    """Calculates the kurtosis of a given signal across all dimensions.

    Args:
        signal (torch.Tensor): Input signal of any shape.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        float: The kurtosis value.
    """
    # 确保使用浮点计算
    signal = signal.float()

    # 将张量展平为一维
    signal_flat = signal.reshape(-1)
    n = signal_flat.shape[0]

    # 计算均值和方差
    mean = torch.mean(signal_flat)
    var = torch.var(signal_flat, unbiased=False)

    # 设置方差的下限，防止除零
    var = max(var, eps)

    # 计算中心化信号
    centered = signal_flat - mean

    # 计算四阶矩
    fourth_moment = torch.sum(centered**4) / n

    # 峰度 = 四阶矩 / 方差的平方
    kurtosis = fourth_moment / (var**2)

    return kurtosis.item()


def calculate_outlier_degree(signal, upper_bound=64, lower_bound=1 / 8):
    """Adapted from PrefixQuant"""
    signal = signal.float().abs()
    token_dim = signal.shape[-1]
    signal = signal.view(-1, token_dim)
    token_max = signal.max(dim=1).values
    all_token_median = token_max.median()
    R = token_max / all_token_median

    res = {}
    res["#upper outliers"] = (R > upper_bound).sum().item()
    res["#lower outliers"] = (R < lower_bound).sum().item()
    res["#total outliers"] = res["#upper outliers"] + res["#lower outliers"]
    res["max outlier degree"] = R.max().item()
    res["min outlier degree"] = R.min().item()
    # res["avg outlier degree"] = R.mean().item()
    # res["outlier degree"] = res["#total outliers"] / token_dim
    return res


def calculate_outlier_degree_channel(signal, eps=1e-8):
    raise NotImplementedError


def draw(save_dir, save_name, X, Y1, Y2):
    """Draws a comparison plot of two signals and saves it.

    Args:
        save_path (str): Path to save the plot.
        save_name (str): Name of the plot.
        X (range): X-axis values.
        Y1 (numpy.ndarray): First signal values.
        Y2 (numpy.ndarray): Second signal values.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y1)
    ax.plot(X, Y2)
    plt.xlabel("channel")
    plt.ylabel("value")
    plt.title(save_name)
    fig.savefig(save_dir / f"{save_name}.jpg")
    plt.close(fig)
    plt.cla()


def block_forward(block, input_data, input_kwargs):
    """Forward pass through a model block.

    Args:
        block (nn.Module): Model block.
        input_data (list): List of input tensors.
        input_kwargs (list): List of kwargs dictionaries for each input.

    Returns:
        list: Output tensors after forward pass.
    """
    output = []

    for i in range(len(input_data)):
        input_data[i] = input_data[i].to(
            device=next(block.parameters()).device,
            dtype=next(block.parameters()).dtype,
        )
        if (
            "attention_mask" in input_kwargs[i]
            and input_kwargs[i]["attention_mask"] is not None
        ):
            input_kwargs[i]["attention_mask"] = input_kwargs[i]["attention_mask"].cuda()
        with torch.no_grad():
            out = block(input_data[i], **input_kwargs[i])
            if out.ndim == 4:
                out = out[0]
            output.append(out)
    return output


def setup_output_dir(output_dir):
    """Creates output directory and sets up logger.

    Args:
        output_dir (str): Path to output directory.

    Returns:
        Path: Path object of the output directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(path / "log.txt", level="INFO", mode="w")

    return path


def get_calib_config(args):
    """Creates calibration configuration from arguments.

    Args:
        args: Command line arguments.

    Returns:
        dict: Calibration configuration.
    """
    return {
        "name": args.dataset_name,
        "download": False,
        "path": args.data_path,
        "n_samples": args.n_samples,
        "bs": args.bs,
        "seq_len": args.seq_len,
        "preproc": args.preproc,
        "seed": args.seed,
    }


def get_model_config(args, model_path):
    """Creates model configuration from arguments.

    Args:
        args: Command line arguments.

    Returns:
        dict: Model configuration.
    """
    return EasyDict(
        {
            "model": {
                "type": args.model_type,
                "path": model_path,
                "torch_dtype": args.torch_dtype,
                "tokenizer_mode": args.tokenizer_mode,
            }
        }
    )


def cosine_similarity(tensor1, tensor2):
    """Calculates cosine similarity between two tensors.

    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Cosine similarity.
    """
    cosine_sim = nn.CosineSimilarity()
    return cosine_sim(tensor1.float().view(1, -1), tensor2.float().view(1, -1))


def calculate_absmax_channel(signal):
    """Calculates the absolute maximum value of a given signal along channels.

    Args:
        signal (torch.Tensor): Input signal of any shape. Calculation will be done
                              along the last dimension.

    Returns:
        float: The average absolute maximum value across all channels.
    """
    # Handle different input shapes
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)

    # Calculate abs max for each channel
    max_values = signal.abs().amax(dim=-1).float()

    # Return the average of max values across all channels
    return torch.mean(max_values).item()


def calculate_absmax(signal):
    """Calculates the absolute maximum value of a given signal across all dimensions.

    Args:
        signal (torch.Tensor): Input signal of any shape.

    Returns:
        float: The absolute maximum value.
    """

    # Calculate absolute maximum
    abs_max = signal.abs().max().float()

    return abs_max.item()
