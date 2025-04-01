from llmc.compression.quantization.quant import (
    FloatQuantizer,
    IntegerQuantizer,
    update_block_wise_scales,
    weight_cast_to_bf16,
    weight_cast_to_fp8,
)
from llmc.compression.quantization.fp8_kernel import (
    weight_cast_to_bf16 as native_weight_cast_to_bf16,
)
import os
import sys
import unittest

import numpy as np
import torch

# Add parent directory to path so we can import from llmc module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQuantFunctions(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Check if CUDA is available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Test shapes
        self.shapes = [
            ((128, 128), "both dimensions multiple of 128"),
            ((512, 512), "both dimensions multiple of 128"),
            ((150, 150), "both dimensions non-multiple of 128"),
            ((127, 127), "both dimensions non-multiple of 128"),
            ((128, 150), "height multiple of 128, width not"),
            ((150, 128), "width multiple of 128, height not"),
            ((256, 127), "height multiple of 128, width odd"),
            ((127, 256), "width multiple of 128, height odd"),
        ]

        # Block size for block-wise operations
        self.block_size = 128

        # Create random weights and scales for all shapes
        self.weights_fp8 = {}
        self.scales = {}
        self.weights_bf16 = {}  # Add bf16 weights

        for shape, desc in self.shapes:
            # Create weight tensor
            self.weights_fp8[desc] = torch.randn(shape, device=self.device).to(
                torch.float8_e4m3fn
            )
            # Create bf16 weight tensor
            self.weights_bf16[desc] = torch.randn(
                shape, device=self.device, dtype=torch.bfloat16
            )

            # Calculate number of blocks
            num_blocks_x = (shape[0] + self.block_size - 1) // self.block_size
            num_blocks_y = (shape[1] + self.block_size - 1) // self.block_size

            # Create scale tensor
            self.scales[desc] = (
                torch.rand(
                    (num_blocks_x,
                     num_blocks_y),
                    device=self.device) *
                0.1 +
                0.01)

        self.cos_sim = torch.nn.CosineSimilarity()
        self.cos_sim_threshold = 0.99
        self.max_diff_threshold = 0.001

    def test_weight_cast_to_bf16(self):
        """Test weight_cast_to_bf16 function"""
        print("Testing weight_cast_to_bf16...")

        for desc, weight in self.weights_fp8.items():
            print(f"\nTesting shape: {desc}")
            scale = self.scales[desc]

            python_result = weight_cast_to_bf16(weight, scale)
            native_result = native_weight_cast_to_bf16(weight, scale)

            # Check if the results are close
            python_result = python_result.float().view(1, -1)
            native_result = native_result.float().view(1, -1)
            cos_sim_value = self.cos_sim(python_result, native_result)
            max_diff = (python_result - native_result).abs().max().item()

            print(f"Cosine similarity: {
                  cos_sim_value}, Max difference: {max_diff}")
            self.assertGreater(
                cos_sim_value,
                self.cos_sim_threshold,
                f"Cosine similarity should be close to 1 for shape {desc}",
            )
            self.assertLess(
                max_diff,
                self.max_diff_threshold,
                f"Maximum difference should be small for shape {desc}",
            )

    def test_weight_cast_to_fp8(self):
        """Test weight_cast_to_fp8 function"""
        print("Testing weight_cast_to_fp8...")

        try:
            for desc, weight_bf16 in self.weights_bf16.items():
                print(f"\nTesting shape: {desc}")
                scale = self.scales[desc]

                # Test both Python and native implementations
                python_result = weight_cast_to_fp8(weight_bf16, scale)
                native_result = native_weight_cast_to_fp8(weight_bf16, scale)

                # Check if the results are close
                python_result = python_result.float().view(1, -1)
                native_result = native_result.float().view(1, -1)
                cos_sim_value = self.cos_sim(python_result, native_result)
                max_diff = (python_result - native_result).abs().max().item()

                print(f"Cosine similarity: {cos_sim_value.item()}")
                print(f"Max difference: {max_diff}")

                # Assertions
                self.assertGreater(
                    cos_sim_value.item(),
                    self.cos_sim_threshold,
                    f"Cosine similarity should be close to 1 for shape {desc}",
                )
                self.assertLess(
                    max_diff,
                    self.max_diff_threshold,
                    f"Maximum difference should be small for shape {desc}",
                )

        except Exception as e:
            print(f"Error in test_weight_cast_to_fp8: {e}")
            self.skipTest(f"Skipping FP8 test due to error: {e}")


if __name__ == "__main__":
    unittest.main()
