import torch
import math
import warnings

def _stabilize_complex_denominator(x, eps):
    """Ensure complex denominators do not collapse to zero."""
    abs_x = torch.abs(x)
    safe_abs = torch.where(abs_x == 0, torch.ones_like(abs_x), abs_x)
    phase = x / safe_abs
    phase = torch.where(abs_x == 0, torch.ones_like(x), phase)
    return torch.where(abs_x < eps, phase * eps, x)

def TMM_solver(thicknesses, refractive_indices, k, theta, pol, debug=False):
    """
    传输矩阵法求解器（支持float64精度和极大n/k值）
    计算多层薄膜的反射率和透射率（支持批处理）

    参数:
        thicknesses: 层厚度张量 (batch_size, num_layers, num_wavelengths) (微米)
        refractive_indices: 折射率张量 (batch_size, num_layers+2, num_wavelengths)
        k: 波数 (1/微米)
        theta: 入射角 (弧度)
        pol: 偏振类型 (0=TE, 1=TM)

    返回:
        R: 反射率 (batch_size, num_wavelengths)
        T: 透射率 (batch_size, num_wavelengths)

    说明:
        - 自动检测输入精度（float32/float64）并使用相应精度计算
        - 支持极大的n、k值（几百到几千）
        - 不对n、k值进行人为限制
        - 当出现数值问题时输出错误信息
    """
    try:
        # 自动检测输入数据类型并确定计算精度
        if refractive_indices.dtype == torch.complex128:
            complex_dtype = torch.complex128
            real_dtype = torch.float64
            eps = 1e-15  # float64的机器精度
        else:
            complex_dtype = torch.complex64
            real_dtype = torch.float32
            eps = 1e-7  # float32的机器精度

        # 确保所有输入使用相同的数据类型
        d = thicknesses.to(complex_dtype) * 1e-6  # 转换为米
        n = torch.permute(refractive_indices, (0, 2, 1))  # 调整折射率张量维度

        # 计算波长
        l = 2 * math.pi / k  # 波长 (μm)
        if isinstance(l, torch.Tensor):
            l = l.to(real_dtype) * 1e-6  # 转换为米
            # 调整l的维度以匹配n: (n_wavelengths,) -> (1, n_wavelengths, 1)
            l = l.view(1, -1, 1)
        else:
            l = l * 1e-6  # 转换为米

        Z0 = 376.730313  # 自由空间阻抗，欧姆

        # 计算传播常数 g = k * n = (2π/λ) * n
        g = 1j * 2 * math.pi * n / l

        # 计算导纳 Y = n / Z0
        Y = n / Z0

        # 计算斯涅尔定律中的 cos(theta_i) = sqrt(1 - (n_0/n_i * sin(theta_0))^2)
        # 替换原有那行
        theta_tensor = torch.as_tensor(theta, dtype=real_dtype, device=n.device)
        sin_theta = torch.sin(theta_tensor)

        sin_theta_squared = sin_theta ** 2

        # 计算 (n_0 / n_i)^2 * sin^2(theta_0)
        # n[:,:,0] 是入射介质的折射率
        sqrt_arg = 1 - ((n[:,:,0:1] / n) ** 2) * sin_theta_squared
        ct = torch.sqrt(sqrt_arg)  # cos(theta) in each layer

        # 计算倾斜导纳 eta
        if pol == 0:
            # TE偏振: eta = Y * cos(theta)
            eta = Y * ct
        else:
            # TM偏振: eta = Y / cos(theta)
            # 添加小量避免除零
            ct_safe = _stabilize_complex_denominator(ct, eps)
            eta = Y / ct_safe

        # 计算相位延迟 delta = k * n * d * cos(theta)
        delta = 1j * g * d * ct

        # 检测delta的范围，对于过大的虚部进行警告
        delta_imag_max = torch.abs(delta.imag).max().item()
        delta_real_max = torch.abs(delta.real).max().item()
        if delta_imag_max > 200 or delta_real_max > 200:
            warnings.warn(
                f"检测到极大的相位延迟: delta.real_max={delta_real_max:.2f}, "
                f"delta.imag_max={delta_imag_max:.2f}。这可能导致数值溢出。"
            )

        # 对于极大的虚部delta（强吸收），限制在合理范围以避免exp溢出
        # exp(1j*delta) = exp(-delta.imag) * exp(1j*delta.real)
        # 当delta.imag < -100时，exp(-delta.imag) > 1e43会溢出
        max_imag = 100.0  # 根据精度调整
        delta = torch.where(
            delta.imag < -max_imag,
            delta.real - max_imag * 1j,
            delta
        )
        delta = torch.where(
            delta.imag > max_imag,
            delta.real + max_imag * 1j,
            delta
        )

        # 计算传输矩阵元素
        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)

        # 初始化传输矩阵
        M = torch.zeros((d.size(0), d.size(1), d.size(2), 2, 2),
                       dtype=complex_dtype, device=d.device)

        # 构建每层的传输矩阵
        # M = [[cos(delta), i*sin(delta)/eta],
        #      [i*eta*sin(delta), cos(delta)]]
        M[:,:,:,0,0] = cos_delta

        # 对于 1/eta，添加小量避免除零
        eta_safe = _stabilize_complex_denominator(eta, eps)
        M[:,:,:,0,1] = 1j * sin_delta / eta_safe
        M[:,:,:,1,0] = 1j * eta * sin_delta
        M[:,:,:,1,1] = cos_delta

        # 计算总传输矩阵（矩阵乘法）
        M_t = torch.zeros((d.size(0), d.size(1), 2, 2),
                         dtype=complex_dtype, device=d.device)
        M_t[:,:,0,0] = 1  # 单位矩阵
        M_t[:,:,1,1] = 1

        # 累积所有层的传输矩阵（去掉首尾的空气层）
        if d.size(-1) > 2:
            M_combined = M[:,:,1:-1,:,:]  # 去掉首尾的空气层
            num_layers = M_combined.size(2)

            if num_layers == 1:
                M_t = M_combined[:,:,0,:,:]
            else:
                # 逐层累积矩阵乘法
                M_t = M_combined[:,:,0,:,:]
                for i in range(1, num_layers):
                    M_t = torch.matmul(M_t, M_combined[:,:,i,:,:])

        # 检查传输矩阵是否包含nan或inf
        if torch.isnan(M_t).any() or torch.isinf(M_t).any():
            warnings.warn(
                f"传输矩阵包含nan或inf值！\n"
                f"nan数量: {torch.isnan(M_t).sum().item()}, "
                f"inf数量: {torch.isinf(M_t).sum().item()}"
            )

        if debug:
            print(f"TMM solver - M_t shape: {M_t.shape}")
            print(f"TMM solver - eta shape: {eta.shape}")

        # 计算反射和透射系数
        # eta[:,:,0] 是入射介质的导纳，eta[:,:,-1] 是出射介质的导纳
        eta_0 = eta[:,:,0]  # 入射介质
        eta_N = eta[:,:,-1]  # 出射介质

        # 计算分母
        denominator = eta_0 * (M_t[:,:,0,0] + M_t[:,:,0,1] * eta_N) + \
                     (M_t[:,:,1,0] + M_t[:,:,1,1] * eta_N)

        # 添加小量避免除零
        denominator_safe = _stabilize_complex_denominator(denominator, eps)

        # 计算反射系数 r
        numerator_r = eta_0 * (M_t[:,:,0,0] + M_t[:,:,0,1] * eta_N) - \
                     (M_t[:,:,1,0] + M_t[:,:,1,1] * eta_N)
        r = numerator_r / denominator_safe

        # 计算透射系数 t
        t = 2 * eta_0 / denominator_safe

        # 计算反射率 R = |r|^2
        R = torch.pow(torch.abs(r), 2)

        # 计算透射率 T = |t|^2 * Re(eta_N / eta_0)
        # 添加小量避免除零
        eta_0_safe = _stabilize_complex_denominator(eta_0, eps)
        T = torch.pow(torch.abs(t), 2) * torch.real(eta_N / eta_0_safe)

        # 检查输出是否包含nan或inf
        if torch.isnan(R).any() or torch.isinf(R).any():
            n_nan_R = torch.isnan(R).sum().item()
            n_inf_R = torch.isinf(R).sum().item()
            warnings.warn(
                f"反射率R包含异常值！nan: {n_nan_R}, inf: {n_inf_R}"
            )

        if torch.isnan(T).any() or torch.isinf(T).any():
            n_nan_T = torch.isnan(T).sum().item()
            n_inf_T = torch.isinf(T).sum().item()
            warnings.warn(
                f"透射率T包含异常值！nan: {n_nan_T}, inf: {n_inf_T}"
            )

        # 确保R和T在物理有效范围内 [0, 1]
        # 但不强制限制，只给出警告
        if (R < -0.01).any() or (R > 1.01).any():
            warnings.warn(
                f"反射率R超出物理范围[0,1]！min={R.min().item():.4f}, max={R.max().item():.4f}"
            )

        if (T < -0.01).any() or (T > 1.01).any():
            warnings.warn(
                f"透射率T超出物理范围[0,1]！min={T.min().item():.4f}, max={T.max().item():.4f}"
            )

        # 检查能量守恒 R + T <= 1（考虑吸收）
        sum_RT = R + T
        if (sum_RT > 1.01).any():
            warnings.warn(
                f"R+T > 1，违反能量守恒！max(R+T)={sum_RT.max().item():.4f}"
            )

        return R, T

    except Exception as e:
        print(f"[ERROR] TMM求解器发生错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
