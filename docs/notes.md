# 本次修改摘要

## 需求
- 阅读 `analysis_optogpt.ipynb` 中的 t-SNE 代码。
- 新增独立 Python 入口，用于对比 SFT 前后两个 OptoGPT checkpoint 的隐藏表示 t-SNE。

## notebook 中 t-SNE 的原始逻辑
- `analysis_optogpt.ipynb`
  - `cell 34`
    - `hidedn_spec = model.fc(hidden_index.to(DEVICE))`
    - 从模型 `fc` 提取光谱隐藏表示
    - 根据 checkpoint 里的 `struc_word_dict` 构造 `mat_index`
  - `cell 35`
    - 把 `hidedn_struc` 与 `hidedn_spec` 拼接后做 `MinMaxScaler + TSNE`
  - `cell 36`
    - 按材料类别和光谱点分别绘制散点图

## 实际修改
- `runners/run_tsne_compare.py`
  - 新增独立 runner：
    - 同时加载 before / after 两个 checkpoint
    - 提取两者的结构 token embedding
    - 提取两者对同一批光谱的 `fc` 隐藏表示
    - 把 before/after 的结构与光谱隐藏表示拼接后做一次共享 t-SNE
    - 输出共享坐标系下的 before/after 双面板对比图
  - 输出：
    - `tsne_compare.png`
    - `tsne_points.npz`
    - `metadata.json`
  - 兼容 `.npy/.pkl` 光谱文件
  - 保持 notebook 里的材料顺序、颜色和图例语义
  - 比 notebook 额外改进：
    - before/after 共用一次 t-SNE 拟合，避免单独拟合导致两张图坐标系不可比
- 修复：
  - 同时处理了 `numpy 2.x` 下 `ndarray.ptp()` 被移除的问题，改为 `np.ptp(...)`

## 运行示例
```bash
python runners/run_tsne_compare.py ^
  --before-checkpoint model/optogpt.pt ^
  --after-checkpoint outputs/sft/<run>/checkpoints/best.pt ^
  --spectrum-path dataset/Spectrum_test.npy ^
  --max-spectra 1000
```

## 验证
- `D:\\anaconda\\envs\\oneday\\python.exe runners/run_tsne_compare.py --help`
- `D:\\anaconda\\envs\\oneday\\python.exe runners/run_tsne_compare.py --before-checkpoint model/optogpt.pt --after-checkpoint model/optogpt.pt --spectrum-path dataset/Spectrum_test.npy --max-spectra 8 --batch-size 8 --n-iter 250 --output-dir outputs/_debug --name tsne_smoke --device auto`

结果：通过

## Git
- branch: `feat/tsne-compare-runner`
- commit: `git commit -m "feat: add tsne comparison runner for sft checkpoints"`
