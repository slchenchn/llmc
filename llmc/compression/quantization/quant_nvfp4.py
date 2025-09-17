from .quant import BaseQuantizer
import torch


class FP8_E4M3_DATA:
    exponent = 4
    mantissa = 3
    bits = 8
    max = torch.finfo(torch.float8_e4m3fn).max
    min = torch.finfo(torch.float8_e4m3fn).min
    min_positive = 0.125
    dtype = torch.float8_e4m3fn

    @staticmethod
    @torch.compile
    def cast_to_fp8(x):
        x = x.clamp(FP8_E4M3_DATA.min, FP8_E4M3_DATA.max)
        return x.to(FP8_E4M3_DATA.dtype)

    @staticmethod
    @torch.compile
    def cast_to_positive_fp8(x):
        x = x.float().clamp(FP8_E4M3_DATA.min_positive, FP8_E4M3_DATA.max)
        return x.to(FP8_E4M3_DATA.dtype)


class FP4_E2M1_DATA:
    exponent = 2
    mantissa = 1
    bits = 4
    max = 6.0
    min = -6.0

    @staticmethod
    @torch.compile
    def cast_to_fp4(x):
        sign = torch.sign(x)
        x = torch.abs(x)
        x[(x >= 0.0) & (x <= 0.25)] = 0.0
        x[(x > 0.25) & (x < 0.75)] = 0.5
        x[(x >= 0.75) & (x <= 1.25)] = 1.0
        x[(x > 1.25) & (x < 1.75)] = 1.5
        x[(x >= 1.75) & (x <= 2.5)] = 2.0
        x[(x > 2.5) & (x < 3.5)] = 3.0
        x[(x >= 3.5) & (x <= 5.0)] = 4.0
        x[x > 5.0] = 6.0
        return x * sign


class NVFP4Quantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.quant_type = "nvfp4-quant"
        assert symmetric, "NVFP4 is always symmetric"
        self.sym = symmetric
        assert self.bit == 4, "Bit must be 4 for NVFP4Quantizer"

        # NVFP4 E2M1 values (matching compressed-tensors reference)
        self.e2m1_to_float = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
        )

        # NVFP4 range and special values
        self.qmin = torch.tensor(-6.0)
        self.qmax = torch.tensor(6.0)
        self.zero_value = 0.0

        # Set up quantization parameters based on granularity
        assert self.granularity == "per_group", (
            "NVFP4Quantizer only supports per_group granularity"
        )
        assert self.group_size == 16, "NVFP4Quantizer only supports group_size=16"

    def get_mse_range(self, tensor, norm=2.4, bs=256):
        raise NotImplementedError

    @staticmethod
    def get_global_scale(global_absmax):
        global_scale = FP8_E4M3_DATA.max * FP4_E2M1_DATA.max / global_absmax.float()

        # global scale is FP32 (positive)
        global_scale = global_scale.view([1]).float()
        return global_scale

    @staticmethod
    def get_local_scales(global_scale, absmax):
        local_scales = global_scale * (absmax / FP4_E2M1_DATA.max)
        # local scale is FP8_E4M3 (positive)
        local_scales = FP8_E4M3_DATA.cast_to_positive_fp8(local_scales)
        return local_scales

    def get_qparams(
        self, tensor_range, device, global_scale=None, min_global_scale=None
    ):
        min_val, max_val = tensor_range[0], tensor_range[1]
        absmax = torch.max(max_val.abs(), min_val.abs())
        # absmax = torch.max(max_val.abs(), min_val.abs()).float()

        global_absmax = absmax.max()
        cur_global_scale = self.get_global_scale(global_absmax)
        if global_scale is None:
            global_scale = cur_global_scale
        else:
            assert cur_global_scale >= global_scale, (
                "cur_global_scale is less than global_scale"
            )
        if min_global_scale is not None:
            assert global_scale >= min_global_scale, (
                "global_scale is less than min_global_scale"
            )
        local_scales = self.get_local_scales(global_scale, absmax).to(device)

        return global_scale, local_scales, self.qmax, self.qmin

    def get_tensor_qparams(self, tensor, args={}, min_global_scale=None):
        """Get quantization parameters for NVFP4"""
        if self.calib_algo == "hqq":
            raise NotImplementedError
            return self.get_hqq_qparams(tensor, args)

        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor, args)
        global_scale, local_scales, qmax, qmin = self.get_qparams(
            tensor_range,
            tensor.device,
            args.get("global_scale", None),
            min_global_scale=min_global_scale,
        )
        return tensor, global_scale, local_scales, qmax, qmin

    def get_batch_tensors_qparams(self, act_tensors, alpha=0.01, args={}):
        if self.calib_algo == "static_hist":
            assert self.sym is True and self.granularity == "per_tensor", (
                "Only support per tensor static symmetric int quantize."
            )
            min_vals, max_vals = self.get_static_hist_range(act_tensors)
        elif self.calib_algo == "static_minmax":
            global_absmax = max(t.abs().max() for t in act_tensors)
            global_scale = self.get_global_scale(global_absmax)
        elif self.calib_algo == "static_moving_minmax":
            raise NotImplementedError
            min_vals, max_vals = self.get_static_moving_minmax_range(act_tensors, alpha)
        else:
            raise ValueError(f"Unsupported calibration algorithm: {self.calib_algo}")

        # for i in range(len(min_vals)):
        #     min_val, max_val = min_vals[i], max_vals[i]
        #     scales, zeros, qmax, qmin = self.get_qparams(
        #         (min_val, max_val), min_val.device
        #     )
        #     scales_list.append(scales)
        #     zeros_list.append(zeros)
        #     qmin_list.append(qmin)
        #     qmax_list.append(qmax)

        # return scales_list, zeros_list, qmin_list, qmax_list
        return (
            [global_scale],
            [torch.tensor([0])],
            [torch.tensor([-6.0])],
            [torch.tensor([6.0])],
        )

    def quant(self, tensor, global_scale, local_scales, qmax, qmin):
        """Quantize tensor to NVFP4 format"""

        # cast local_scales back to fp32, to accommodate old hardwares
        local_scales = FP8_E4M3_DATA.cast_to_positive_fp8(local_scales)

        x_scaled = (tensor / local_scales.float()) * global_scale
        x_quant = FP4_E2M1_DATA.cast_to_fp4(x_scaled)

        return x_quant

    def dequant(self, tensor, global_scale, local_scales):
        """Dequantize from NVFP4 format"""

        # Apply inverse scaling
        dequantized = tensor.float() * local_scales.float() / global_scale

        return dequantized

    def quant_dequant(self, tensor, global_scale, local_scales, qmax, qmin):
        """Perform quantization and dequantization (simulates the quantization effect)"""
        tensor = self.quant(tensor, global_scale, local_scales, qmax, qmin)
        tensor = self.dequant(tensor, global_scale, local_scales)
        return tensor

    def fake_quant_weight_static(self, weight, args):
        """Fake quantization for static weights"""
        if "dim" in args and "ic" in args["dim"]:
            qweight = weight.T
        else:
            qweight = weight

        org_w_shape = qweight.shape
        org_w_dtype = qweight.dtype

        assert "output_scale_factor" not in args, (
            "output_scale_factor is not supported for NVFP4Quantizer"
        )
        global_scale, local_scales, qmax, qmin = (
            args["global_scale"],
            args["local_scales"],
            args["qmax"],
            args["qmin"],
        )

        qweight = self.reshape_tensor(qweight)
        qweight = self.quant_dequant(qweight, global_scale, local_scales, qmax, qmin)
        qweight = self.restore_tensor(qweight, org_w_shape).to(org_w_dtype)

        if "dim" in args and "ic" in args["dim"]:
            qweight = qweight.T

        return qweight

    def fake_quant_weight_dynamic(self, weight, args={}):
        """Fake quantization for dynamic weights"""
        raise NotImplementedError
        if "dim" in args and "ic" in args["dim"]:
            q_weight = weight.T
        else:
            q_weight = weight

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype

        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if "dim" in args and "ic" in args["dim"]:
            q_weight = q_weight.T

        return q_weight

    def real_quant_weight_static(self, weight, args):
        org_w_shape = weight.shape
        if "output_scale_factor" in args:
            output_scale_factor = args["output_scale_factor"]
            del args["output_scale_factor"]
        else:
            output_scale_factor = 1

        global_scale, local_scales, qmax, qmin = (
            args["global_scale"],
            args["local_scales"],
            args["qmax"],
            args["qmin"],
        )
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, global_scale, local_scales, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        local_scales = local_scales.float() * output_scale_factor
        local_scales = FP8_E4M3_DATA.cast_to_positive_fp8(local_scales)
        local_scales = local_scales.view(org_w_shape[0], org_w_shape[1] // 16)

        assert len(weight.unique()) <= 15
        assert global_scale.dtype == torch.float32
        assert global_scale.numel() == 1
        assert local_scales.dtype == torch.float8_e4m3fn
        return weight, global_scale, local_scales

    def real_quant_weight_dynamic(self, weight, args={}):
        """Real quantization for dynamic weights (returns encoded NVFP4 and scales).

        - Returns uint8-typed tensor containing NVFP4 4-bit codes (one per element).
        - Returns scales shaped per granularity (per_group for NVFP4).
        - Zeros is None since NVFP4 is symmetric.
        """
        org_w_shape = weight.shape

        # Optional post scaling factor for output scales (e.g., to fold into kernel)
        if "output_scale_factor" in args:
            output_scale_factor = args["output_scale_factor"]
            # do not leak into sub-calls that may inspect args
            del args["output_scale_factor"]
        else:
            output_scale_factor = 1

        # Get per-tensor/per-group params and quantize to NVFP4 codes
        weight, global_scale, local_scales, qmax, qmin = self.get_tensor_qparams(
            weight, args, min_global_scale=100
        )
        weight = self.quant(weight, global_scale, local_scales, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        local_scales = local_scales.float() * output_scale_factor
        local_scales = FP8_E4M3_DATA.cast_to_positive_fp8(local_scales)
        local_scales = local_scales.view(org_w_shape[0], org_w_shape[1] // 16)

        assert len(weight.unique()) <= 15
        assert global_scale.dtype == torch.float32
        assert global_scale.numel() == 1
        assert local_scales.dtype == torch.float8_e4m3fn
        return weight, global_scale, local_scales

    def fake_quant_act_static(self, act, args={}):
        org_act_shape = act.shape
        org_act_dtype = act.dtype

        global_scale, zeros, qmax, qmin = (
            args["scales"],
            args["zeros"],
            args["qmax"],
            args["qmin"],
        )
        assert torch.all(zeros == 0), f"NVFP4Quantizer's zeros must be 0, got {zeros}"
        assert torch.all(qmax == 6.0), f"NVFP4Quantizer's qmax must be 6.0, got {qmax}"
        assert torch.all(qmin == -6.0), (
            f"NVFP4Quantizer's qmin must be -6.0, got {qmin}"
        )

        act = self.reshape_tensor(act)
        local_scales = self.get_tensor_qparams(act, {"global_scale": global_scale})[2]
        act = self.quant_dequant(act, global_scale, local_scales, qmax, qmin)
        act = self.restore_tensor(act, org_act_shape).to(org_act_dtype)

        return act

    def fake_quant_act_dynamic(self, act, args={}):
        raise NotImplementedError
