"""把现有 pkl 数据转换为更易重复读取的 npy 文件。"""

from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

import numpy as np


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 Spectrum/Structure 的 pkl 文件转换成 npy。")
    parser.add_argument("--input", required=True, help="输入 pkl 文件路径")
    parser.add_argument("--output", required=True, help="输出 npy 文件路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_pickle(input_path)
    # 结构数据是变长 token 列表，这里会保存成 object 数组；
    # 光谱数据通常会保存成 float32 二维数组。
    if isinstance(data, np.ndarray):
        array = data
    else:
        array = np.asarray(data, dtype=object if data and isinstance(data[0], (list, tuple, str)) else None)
    np.save(output_path, array, allow_pickle=True)
    print(f"已转换: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
