#!/usr/bin/env python3
"""
An advanced test suite for comparing our NVFP4Quantizer against the
`compressed-tensors` reference implementation.

This suite covers:
1.  Tensor-wise quantization with various matrix shapes.
2.  Group-wise quantization (local scales).
3.  Application of a global scale parameter.
"""

import torch
import sys
import traceback
import unittest

# Add project root to path
sys.path.append('.')

# --- Import necessary components ---
try:
    # Our implementation
    from llmc.compression.quantization.quant import NVFP4Quantizer
    # Reference implementation from compressed-tensors
    from compressed_tensors.quantization.quant_args import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
    COMPRESSED_TENSORS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import a required library: {e}", file=sys.stderr)
    COMPRESSED_TENSORS_AVAILABLE = False


def compare_quantized_outputs(
    test_case, our_output, ref_output, msg, tolerance=1e-6
):
    """Helper function to compare two tensors and assert their equality."""
    test_case.assertTrue(
        torch.allclose(our_output, ref_output, atol=tolerance),
        f"{msg}: Tensors do not match. Max diff: "
        f"{torch.max(torch.abs(our_output - ref_output)).item()}"
    )

@unittest.skipIf(not COMPRESSED_TENSORS_AVAILABLE, "compressed-tensors library not available")
class TestNVFP4QuantizerAdvanced(unittest.TestCase):

    def setUp(self):
        """Set up common objects for tests."""
        self.our_quantizer = NVFP4Quantizer(bit="nvfp4", symmetric=True, granularity="per_tensor")
        self.ref_args = QuantizationArgs(num_bits=4, type="float")

    def _run_comparison(self, tensor, our_quantizer, ref_args, global_scale=None):
        """Helper to run a full quantization comparison."""
        
        # --- Our Implementation ---
        # `fake_quant_weight_dynamic` handles scale calculation and quant-dequant
        our_args = {"global_scale": global_scale}
        our_dequantized = our_quantizer.fake_quant_weight_dynamic(tensor, args=our_args)

        # --- Reference Implementation ---
        # 1. Calculate scales and zero points
        #    Note: ref implementation needs a dummy module for some reason
        dummy_module = torch.nn.Module()
        ref_scale, ref_zp = compute_dynamic_scales_and_zp(tensor, ref_args, module=dummy_module)

        # 2. Perform fake quantization
        ref_dequantized = fake_quantize(tensor, ref_scale, ref_zp, ref_args, global_scale=global_scale)
        
        return our_dequantized, ref_dequantized

    def test_tensorwise_quantization_various_shapes(self):
        """Test tensor-wise quantization with different matrix shapes."""
        print("\n--- Testing Tensor-wise Quantization (Various Shapes) ---")
        shapes = [(64, 128), (128, 64), (1, 256), (257, 1), (31, 63)]
        for shape in shapes:
            with self.subTest(shape=shape):
                print(f"Testing shape: {shape}")
                tensor = torch.randn(shape) * 2  # Scaled random data
                
                our_result, ref_result = self._run_comparison(
                    tensor, self.our_quantizer, self.ref_args
                )
                
                compare_quantized_outputs(
                    self, our_result, ref_result, f"Tensor-wise comparison for shape {shape}"
                )
        print("✅ SUCCESS: Tensor-wise quantization matches.")

    def test_groupwise_quantization_local_scales(self):
        """Test group-wise quantization with local scales."""
        print("\n--- Testing Group-wise Quantization (Local Scales) ---")
        shape = (128, 256)
        group_size = 64
        
        our_quantizer_group = NVFP4Quantizer(
            bit="nvfp4", symmetric=True, granularity="per_group", group_size=group_size
        )
        ref_args_group = QuantizationArgs(
            num_bits=4, type="float", strategy="group", group_size=group_size
        )
        
        tensor = torch.randn(shape) * 5
        
        our_result, ref_result = self._run_comparison(
            tensor, our_quantizer_group, ref_args_group
        )
        
        compare_quantized_outputs(
            self, our_result, ref_result, "Group-wise comparison"
        )
        print("✅ SUCCESS: Group-wise quantization matches.")

    def test_global_scale_application(self):
        """Test the application of a global scale factor."""
        print("\n--- Testing Global Scale Application ---")
        shape = (64, 128)
        global_scale = torch.tensor(2.5) # An arbitrary float
        
        tensor = torch.randn(shape) * 3
        
        our_result, ref_result = self._run_comparison(
            tensor, self.our_quantizer, self.ref_args, global_scale=global_scale
        )
        
        compare_quantized_outputs(
            self, our_result, ref_result, "Global scale comparison"
        )
        print("✅ SUCCESS: Global scale application matches.")


if __name__ == "__main__":
    print("--- Running Advanced NVFP4 Quantizer Tests ---")
    unittest.main()
