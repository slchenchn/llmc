from aihub.client import Client
import os
import argparse
from pathlib import Path

BASE = "http://192.168.11.18:30021"
# token = os.getenv("AI_HUB_TOKEN")
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjI4NTc4MjYsImlhdCI6MTc2MjI1MzAyNiwidWlkIjo0MX0.0KTgFLJ7MwJUpVRHKLafdytGvvJEDd8cJPlZFgZmws8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", type=str, choices=["ultrachat"])
    parser.add_argument(
        "--local_root",
        default="data",
        type=str,
        help="本地缓存目录",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ds_cfg = {
        "ultrachat": "ultrachat/V2",
    }
    data_name_list = args.data_names.split(",")
    for data_name in data_name_list:
        assert data_name in ds_cfg.keys(), f"{data_name} is not in {ds_cfg.keys()}"
    print(f"{data_name_list=}")

    local_root = Path(args.local_root)
    with Client(base_url=BASE, token=token) as cli:
        # 2. 下载数据集
        for data_name in data_name_list:
            print(f"{data_name=}")
            dataname_aihub = ds_cfg[data_name]
            # local_dir = local_root / ds_cfg[data_name]
            # local_dir.mkdir(parents=True, exist_ok=True)
            cli.dataset_management.run_download(
                dataset_version_name=dataname_aihub, local_dir=local_root, worker=4
            )


if __name__ == "__main__":
    main()
