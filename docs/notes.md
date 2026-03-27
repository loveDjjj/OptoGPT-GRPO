# Notes

## 需求
将 `dataset/Structure_train.pkl`、`dataset/Structure_test.pkl`、`dataset/Spectrum_train.pkl`、`dataset/Spectrum_test.pkl`
转换为 `.npy`，检查当前代码是否适配 `.npy`，并分析/优化当前多卡并行与训练加速路径。

## 修改文件
- README.md
- configs/eval/spectrum_eval.yaml
- configs/sft/spectral_sft.yaml
- datasets/optogpt_dataset.py
- datasets/splits.py
- scripts/pkl_to_npy.py
- evaluators/spectrum_evaluator.py
- losses/spectrum_loss.py
- trainers/spectral_sft_trainer.py
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 已将四个数据文件转换为 `.npy`：
  - `dataset/Spectrum_train.npy`
  - `dataset/Spectrum_test.npy`
  - `dataset/Structure_train.npy`
  - `dataset/Structure_test.npy`
- 数据加载层保留 `.pkl/.npy` 双兼容，并对数值型 `.npy` 优先使用 `mmap_mode='r'`。
- `scripts/pkl_to_npy.py` 会将光谱数组压缩为 `float32`，降低磁盘与多卡重复读取开销。
- 评测与训练配置默认切换到 `.npy` 路径。
- 补充 `bucket_by_layer_count`、`persistent_workers`、`prefetch_factor` 和训练阶段 `DDP no_sync()`，减少 TMM padding 浪费与无效梯度同步。
- 新增 `data.skip_train_structure_loading`，让 spectral SFT 训练阶段只读取 `Spectrum_train.npy`，避免每个 rank 重复加载超大的 `Structure_train.npy`。
- README 已同步修正为 `.npy` 数据事实状态，并补充 Windows/Gloo 与 Linux/NCCL 的多卡说明。

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -c "import json; from datasets import build_split_datasets; from utils.config import load_yaml_config; cfg=load_yaml_config(r'configs/sft/spectral_sft.yaml'); cfg['data']['max_train_samples']=2; cfg['data']['max_val_samples']=2; ds=build_split_datasets(cfg); report={'train_len': len(ds['train']), 'val_len': len(ds['val']), 'train_structure_loaded': ds['train'].raw_structure_store is not None, 'val_structure_loaded': ds['val'].raw_structure_store is not None, 'train_sample0_structure_len': len(ds['train'][0]['structure_tokens']), 'val_sample0_structure_len': len(ds['val'][0]['structure_tokens']), 'train_sample0_spectrum_dtype': str(ds['train'][0]['spectrum'].dtype), 'val_sample0_spectrum_dtype': str(ds['val'][0]['spectrum'].dtype)}; print(json.dumps(report, ensure_ascii=False))"
```

结果：
- `train_structure_loaded=false`
- `val_structure_loaded=true`
- 训练与验证样本光谱 dtype 均为 `float32`

```bash
D:\anaconda\envs\oneday\python.exe -c "import compileall; paths=['runners','models','datasets','evaluators','losses','physics','trainers','utils','scripts','core']; ok=True\nfor path in paths:\n    result=compileall.compile_dir(path, quiet=1)\n    print(f'{path}: {result}')\n    ok = ok and result\nprint(f'overall: {ok}')"
```

结果：通过

## Git
- branch: `feat/npy-loader-and-ddp-io-opt`
- commit: `git commit -m "feat: add npy dataset support and reduce ddp training io"` 
