# Notes

## 需求
将 SFT 与光谱测评中的光谱损失从“吸收率误差”改为直接比较 `R/T` 光谱的误差，不再默认使用吸收率损失。

## 修改文件
- physics/spectrum.py
- losses/spectrum_loss.py
- configs/eval/spectrum_eval.yaml
- configs/sft/spectral_sft.yaml
- README.md
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- 将光谱误差默认指标从 `absorption_rmse` 改为 `rt_rmse`。
- `rt_rmse` 直接比较拼接后的 `[R..., T...]` 光谱 RMSE，符合当前“直接看 R/T，不看吸收率”的需求。
- 为避免旧配置失效，代码中仍保留 `absorption_rmse` 兼容路径。
- 额外显式支持：
  - `r_rmse`
  - `t_rmse`
  - `rt_mae`
- 评测与 SFT 配置都已切到 `rt_rmse`。
- README 已同步说明当前默认损失是 `R/T` 直接误差。

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -c "from physics.spectrum import spectrum_error; import numpy as np; x=np.arange(142,dtype=np.float32); print('rt_rmse', spectrum_error(x,x,metric='rt_rmse')); print('r_rmse', spectrum_error(x,x,metric='r_rmse')); print('t_rmse', spectrum_error(x,x,metric='t_rmse')); print('absorption_rmse', spectrum_error(x,x,metric='absorption_rmse'))"
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
- branch: `feat/rt-spectrum-loss`
- commit: `git commit -m "feat: switch spectral loss to rt error"`
