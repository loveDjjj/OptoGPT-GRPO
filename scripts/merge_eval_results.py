"""合并多卡评测产生的分片 JSONL。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 outputs/eval 下的多卡样本分片结果。")
    parser.add_argument("--input-dir", required=True, help="样本分片目录，例如 outputs/eval/.../samples")
    parser.add_argument("--pattern", default="*.jsonl", help="需要合并的文件匹配模式")
    parser.add_argument("--output", required=True, help="合并后的输出文件")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(input_dir.glob(args.pattern))
    if not shard_paths:
        raise FileNotFoundError(f"在 {input_dir} 下未找到匹配 {args.pattern} 的分片文件。")

    with output_path.open("w", encoding="utf-8") as writer:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    writer.write(line)

    print(f"已合并 {len(shard_paths)} 个分片文件到: {output_path}")


if __name__ == "__main__":
    main()
