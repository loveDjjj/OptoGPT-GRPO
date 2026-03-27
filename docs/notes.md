# Notes

## 需求
修复 `python runners/run_spectrum_eval.py --config ...` 直接运行时的导入错误，并补充光谱评测阶段的可视化图片输出。

## 修改文件
- runners/run_spectrum_eval.py
- runners/run_spectral_sft.py
- evaluators/spectrum_evaluator.py
- utils/plotting.py
- configs/eval/spectrum_eval.yaml
- README.md
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 为两个 runner 增加仓库根目录引导，允许直接使用 `python runners/...` 运行，避免 `ModuleNotFoundError: No module named 'datasets'`。
- 为评测器新增可选的光谱对比图保存逻辑，图中同时绘制目标/预测的 `R/T/A` 三条曲线。
- 在评测配置中新增：
  - `evaluation.save_plots`
  - `evaluation.plot_max_samples`
- 默认开启评测出图，并限制每个 rank 最多保存 32 张图，避免全量评测时生成过多图片。
- README 已同步补充评测图像输出目录。

## 验证
```bash
D:\anaconda\envs\oneday\python.exe runners/run_spectrum_eval.py --help
D:\anaconda\envs\oneday\python.exe runners/run_spectral_sft.py --help
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
- branch: `fix/runner-import-and-eval-plots`
- commit: `git commit -m "fix: support direct runners execution and eval plots"`
