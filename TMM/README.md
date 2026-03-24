# TMM 使用说明（批量计算 / 单结构计算 / 梯度回传）

本目录是一个基于 `PyTorch` 的多层薄膜传输矩阵法（TMM）实现，支持：

- 从 `database/*.xlsx` 读取材料的色散数据（`n(λ), k(λ)`）
- 批量结构并行计算（CPU/GPU 自动选择）
- 单结构计算（把 batch 大小设为 1）
- 训练友好的高层接口（返回 torch 张量，可选保留梯度）
- 直接调用底层求解器进行 `autograd` 反向传播

---

## 1. 目录结构

- `optical_calculator.py`
  - 高层接口：材料读取、插值、缓存、批量封装、调用求解器
- `TMM.py`
  - 底层 `TMM_solver`：核心矩阵计算逻辑
- `database/*.xlsx`
  - 材料数据库（前三列分别为：波长(um)、n实部、k虚部）
- `demo.py`
  - 演示脚本：单结构 / 批量 / 梯度回传

---

## 2. 依赖环境

建议 Python 3.10+，主要依赖：

- `numpy`
- `torch`
- `pandas`
- `scipy`
- `matplotlib`（仅绘图需要）

示例安装：

```bash
pip install numpy torch pandas scipy matplotlib
```

---

## 3. 快速开始

### 3.1 运行完整 demo

```bash
python demo.py --mode all
```

默认会在 `demo_outputs/` 生成：

- 单结构谱线（CSV + 图）
- 批量结果（NPZ + 图 + summary）
- autograd 梯度示例结果（JSON）

### 3.2 只跑单结构

```bash
python demo.py --mode single
```

### 3.3 只跑批量

```bash
python demo.py --mode batch --batch-size 100 --num-layers 10
```

### 3.4 只跑梯度演示

```bash
python demo.py --mode autograd
```

---

## 3.5 参考结果展示

以下是 `reference_result` 目录中的对比分析图：

### 最差样本曲线对比

![worst_case_curve_compare](reference_result/worst_case_curve_compare.png)

### 波长维度误差

![error_vs_wavelength](reference_result/error_vs_wavelength.png)

### 误差与入射角关系

![error_vs_angle](reference_result/error_vs_angle.png)

### 误差分布直方图

![error_histogram](reference_result/error_histogram.png)

### 按偏振分组误差箱线图

![error_boxplot_by_polarization](reference_result/error_boxplot_by_polarization.png)

---

## 4. API 调用方法

### 4.1 高层批量接口（推荐）

入口：`calculate_optical_properties_batch`（`optical_calculator.py`）

```python
from TMM import calculate_optical_properties_batch, calculate_absorption
import numpy as np

structures = [
    {
        "materials": ["SiO2", "Ge", "SiO2"],
        "thicknesses": [0.45, 0.18, 0.30],  # um
        "database_path": "database",
    }
]

wavelengths, R, T = calculate_optical_properties_batch(
    structures,
    wavelength_range=(3.0, 8.0),      # um
    num_points=500,
    incident_angle=np.deg2rad(15.0),  # rad
    polarization=0,                   # 0=TE(s), 1=TM(p)
)
A = calculate_absorption(wavelengths, R, T)
```

返回：

- `wavelengths.shape == (num_points,)`
- `R.shape == (batch_size, num_points)`
- `T.shape == (batch_size, num_points)`

> 单结构计算 = `batch_size=1` 的批量计算。

---

### 4.2 训练友好高层接口（推荐）

入口：`calculate_optical_properties_batch_torch`（`optical_calculator.py`）

这个接口与 `calculate_optical_properties_batch` 的主要区别：

- 返回 `torch.Tensor`，不转 `numpy`
- 支持 `keep_grad=True`
- `thicknesses` 可直接传入 `requires_grad=True` 的 `torch.Tensor`

示例：

```python
import numpy as np
import torch
from TMM import calculate_optical_properties_batch_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thickness = torch.tensor([0.35, 0.22, 0.18], dtype=torch.float64, device=device, requires_grad=True)

structures = [{
    "materials": ["SiO2", "Al2O3", "HfO2"],
    "thicknesses": thickness,
    "database_path": "database",
}]

wavelengths_t, R_t, T_t = calculate_optical_properties_batch_torch(
    structures,
    wavelength_range=(3.0, 8.0),
    num_points=500,
    incident_angle=torch.tensor(np.deg2rad(20.0), dtype=torch.float64, device=device),
    polarization=0,
    device=device,
    complex_dtype=torch.complex128,
    keep_grad=True,
)

mask = (wavelengths_t >= 4.8) & (wavelengths_t <= 5.2)
loss = -R_t[0, mask].mean()
loss.backward()
print(thickness.grad)
```

---

### 4.3 底层接口（完全自定义）

入口：`TMM_solver`（`TMM.py`）

你需要自己准备好：

- `thicknesses`: `(batch, wl_len, num_layers+2)`
- `refractive_indices`: `(batch, num_layers+2, wl_len)`
- `k`: `(wl_len,)`，`k = 2π/λ`

`demo.py` 的 `autograd_demo` 给了完整可运行例子。

---

## 5. 底层实现说明（核心流程）

`TMM_solver` 的主要计算链路：

1. 根据输入数据类型选择精度（`complex64/complex128`）
2. 计算各层角度项 `cos(theta_i)`（复数域）
3. 计算倾斜导纳 `eta`
4. 计算相位项 `delta`
5. 构建每层 2x2 传输矩阵并连乘得到总矩阵 `M_t`
6. 由边界条件求 `r/t`，再得到 `R/T`

其中高层接口会自动：

- 加空气入射层与出射层（`n=1`，厚度0）
- 从数据库读取并插值 `n,k`
- 利用缓存减少重复读取和插值开销

---

## 6. 数值稳定性与极小值处理

你在对比第三方 `tmm` 时看到“极不透明层会被改成微小透射”提示；我们当前实现的策略不同，主要通过以下方式稳定数值：

1. **相位虚部截断**
   - 在 `TMM.py` 中对 `delta.imag` 限制到 `[-100, 100]`
   - 避免三角/指数相关运算溢出或下溢

2. **复数分母稳定器**
   - `_stabilize_complex_denominator(x, eps)`
   - 当分母模长过小，用同相位的 `eps` 替代，避免除零/爆炸

3. **异常检测**
   - 对 `nan/inf`、`R/T` 越界、`R+T>1` 给出 warning

这意味着：

- 我们没有像某些库那样把透过率强制抬到固定下限（如 `1e-30`）
- 但会通过截断和分母保护来保证数值可计算

---

## 7. 梯度回传（重点）

### 7.1 当前高层接口默认**不保留梯度**

`calculate_optical_properties_batch` 内部使用了：

- `with torch.no_grad():`
- 输出转 `numpy`

因此这个接口是“推理/批量计算”模式，不用于反向传播。

### 7.2 训练友好高层接口（推荐）

如果你希望保留高层封装（自动读取数据库、自动组批）同时支持反向传播，请使用：

- `calculate_optical_properties_batch_torch(..., keep_grad=True)`

并满足：

1. 需要优化的参数（如层厚）用 `torch.Tensor(..., requires_grad=True)` 传入 `thicknesses`
2. 不要把输出转成 `numpy` 再求 loss
3. 用 torch 的 loss 调 `loss.backward()`

完整例子见 `demo.py` 的 `autograd_demo`。

### 7.3 直接底层反向传播（高级用法）

请直接使用 `TMM_solver`，并满足：

1. 不使用 `torch.no_grad()`
2. 参数（例如层厚）设置 `requires_grad=True`
3. 保持张量在 `torch` 中，不要转 `numpy`

示例（简化）：

```python
thickness = torch.tensor([0.35, 0.22, 0.18], dtype=torch.float64, requires_grad=True)
# ... 组装 thicknesses/refractive_indices/k ...
R, T = TMM_solver(thicknesses, refractive_indices, k, theta, pol=0)
loss = -R[:, target_idx].mean()
loss.backward()
print(thickness.grad)
```

完整版本请看 `demo.py` 的 `autograd_demo`（当前演示使用的是训练友好高层接口）。

---

## 8. demo.py 详细说明

`demo.py` 支持 4 种模式：

- `--mode single`
  - 用 1 个结构演示高层接口调用，输出 `R/T/A` 曲线
- `--mode batch`
  - 随机生成多结构，演示批量并行
- `--mode autograd`
  - 使用 `calculate_optical_properties_batch_torch` 计算并反传厚度梯度
- `--mode all`
  - 依次运行以上全部

常用参数：

- `--wl-min --wl-max --num-points`
- `--batch-size --num-layers`
- `--seed`
- `--output-dir`

---

## 9. 设计约束与注意事项

1. 批量接口要求同一批次结构的层数一致  
2. 批量接口默认入射/出射介质固定为空气（`n=1`）  
3. 材料名需要与 `database/*.xlsx` 文件名一致（不含后缀）  
4. `database_path` 推荐统一传 `"database"`  
5. 当前 `TMM_solver` 中保留了调试 `print`（矩阵形状），大批量时会有较多日志

---

## 10. 建议工作流

1. 先用 `demo.py --mode single` 验证材料和单位  
2. 再用 `demo.py --mode batch` 跑大样本  
3. 若要做反向设计/训练，切到 `--mode autograd` 参考训练友好高层接口调用方式  

---
