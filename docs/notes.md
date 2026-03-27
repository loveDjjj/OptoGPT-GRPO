# Notes

## 需求
在训练和评测阶段增加按 DataLoader batch 数量统计的进度条显示。

## 修改文件
- trainers/spectral_sft_trainer.py
- evaluators/spectrum_evaluator.py
- configs/sft/spectral_sft.yaml
- configs/eval/spectrum_eval.yaml
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 训练阶段加入 `tqdm` 进度条，按每个 epoch 的 batch 总数显示。
- 评测阶段加入 `tqdm` 进度条，按每个 split 的 batch 总数显示。
- 只在主进程显示进度条，避免多卡时所有 rank 一起刷屏。
- 训练进度条会按 `log_interval` 刷新最新的 `objective / spectrum / valid`。
- 评测进度条会滚动显示当前累计的 `seq / spec / samples`。
- 在 YAML 中增加 `logging.show_progress_bar: true`，默认开启。

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -c "import tqdm; print(tqdm.__version__)"
```

结果：通过

```bash
D:\anaconda\envs\oneday\python.exe -c "import compileall; paths=['runners','models','datasets','evaluators','losses','physics','trainers','utils','scripts','core']; ok=True
for path in paths:
    result=compileall.compile_dir(path, quiet=1)
    print(f'{path}: {result}')
    ok = ok and result
print(f'overall: {ok}')"
```

结果：通过

## Git
- branch: `feat/batch-progress-bars`
- commit: `git commit -m "feat: add batch-based tqdm progress bars"`
