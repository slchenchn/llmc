import os
import argparse
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="使用llmc进行模型的量化")
    parser.add_argument("--model_path", required=True, type=str, help="模型路径")
    parser.add_argument("--output_dir", required=True, type=str, help="输出模型的路径")
    args = parser.parse_args()
    return args


def get_config(args, src_yml_path, dst_yml_path):
    with open(src_yml_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    model_path = Path(args.model_path) / "before_gptq"
    if not model_path.exists():
        model_path = model_path.parent
        assert model_path.exists(), "模型路径不存在"
    config["model"]["path"] = str(model_path)
    config["save"]["save_path"] = args.output_dir
    print("config:")
    print(config)

    with open(dst_yml_path, "w") as fp:
        yaml.dump(config, fp, allow_unicode=True)

    return dst_yml_path


def main():
    args = parse_args()
    if "rotate_tp16" in args.output_dir:
        src_yml_path = Path(
            "configs/csl/workflow/ostquant_gptqv2_qwen3_sym_w4a8_force_dtype_fp16_rotate_tp16.yml"
        )
    else:
        src_yml_path = Path(
            "configs/csl/workflow/ostquant_gptqv2_qwen3_sym_w4a8_force_dtype_fp16.yml"
        )
    llmc_dir = Path(".")

    dst_yml_path = Path("configs/csl/workflow/tmp.yml")
    get_config(args, src_yml_path, llmc_dir / dst_yml_path)

    # output_dir is the log directory
    os.system(f"cd {llmc_dir};bash scripts/csl_run.sh {dst_yml_path} {args.output_dir}")


if __name__ == "__main__":
    main()
