import os, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def plot_Tstar_multimodel(outcomes, Tc_collapse=None, nu_collapse=None, d=2, output_dir=None):
    """
    自动进行三种相变模型的外推拟合，并分别绘制三张图。
    
    参数:
        outcomes: 字典 {L: {'temps': list, 'ids': list}}
        Tc_collapse: 从数据坍塌得到的 Tc（用于搜索极小值，以及作为参考线）
        nu_collapse: 二级相变所需的指数（若无则跳过二级拟合）
        d: 一级相变的空间维度（默认2）
        output_dir: 输出目录（若为 None 则只显示不保存）
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 第一步：提取每个 L 的 T*(L)（局部极小值温度）
    L_vals = []
    T_star_vals = []
    for L, data in outcomes.items():
        temps = np.array(data['temps'])
        ids = np.array(data['ids'])
        if Tc_collapse is not None:
            # 在 Tc_collapse 附近搜索极小值（半宽 0.1）
            mask = (temps >= Tc_collapse - 0.1) & (temps <= Tc_collapse + 0.1)
            if mask.sum() < 3:
                # 扩大范围
                mask = (temps >= Tc_collapse - 0.2) & (temps <= Tc_collapse + 0.2)
            if mask.sum() >= 3:
                idx_min = np.argmin(ids[mask])
                T_star = temps[mask][idx_min]
                L_vals.append(L)
                T_star_vals.append(T_star)
                continue
        # 如果没有 Tc_collapse 或窗口内点不足，则全局找极小值（谨慎）
        idx_min = np.argmin(ids)
        T_star = temps[idx_min]
        L_vals.append(L)
        T_star_vals.append(T_star)

    if len(L_vals) < 3:
        print("有效 L 数量不足，无法进行外推拟合")
        return

    L_vals = np.array(L_vals)
    T_star_vals = np.array(T_star_vals)
    results = {}

    # ---------- 模型1：二级相变 (需要 nu_collapse) ----------
    if nu_collapse is not None and nu_collapse > 0:
        x = L_vals ** (-1.0 / nu_collapse)
        coeffs = np.polyfit(x, T_star_vals, 1)
        a, b = coeffs
        Tc_extrap = b
        x_fit = np.linspace(0, max(x), 50)
        y_fit = a * x_fit + b
        results['second'] = {'Tc': Tc_extrap, 'a': a, 'b': b}
        # 绘图
        plt.figure(figsize=(6,5))
        plt.plot(x, T_star_vals, 'o', color='black', label='Data')
        plt.plot(x_fit, y_fit, 'r--', label=f'Linear fit, Tc = {Tc_extrap:.4f}')
        if Tc_collapse is not None:
            plt.axhline(y=Tc_collapse, color='blue', linestyle=':', label=f'Tc from collapse = {Tc_collapse:.4f}')
        plt.xlabel(r'$L^{-1/\nu}$')
        plt.ylabel(r'$T^*(L)$')
        plt.grid(alpha=0.3)
        plt.legend()
        if output_dir:
            plt.savefig(f"{output_dir}/Tstar_second.pdf", bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    else:
        print("未提供 nu_collapse，跳过二级相变拟合")

    # ---------- 模型2：BKT 相变 ----------
    x = 1.0 / (np.log(L_vals) ** 2)
    coeffs = np.polyfit(x, T_star_vals, 1)
    a, b = coeffs
    Tc_extrap = b
    results['bkt'] = {'Tc': Tc_extrap, 'a': a, 'b': b}
    # 绘图
    plt.figure(figsize=(6,5))
    plt.plot(x, T_star_vals, 'o', color='black', label='Data')
    plt.plot(np.linspace(0, max(x), 50), a * np.linspace(0, max(x), 50) + b, 'r--', label=f'BKT fit, Tc = {Tc_extrap:.4f}')
    if Tc_collapse is not None:
        plt.axhline(y=Tc_collapse, color='blue', linestyle=':', label=f'Tc from collapse = {Tc_collapse:.4f}')
    plt.xlabel(r'$1/(\ln L)^2$')
    plt.ylabel(r'$T^*(L)$')
    plt.grid(alpha=0.3)
    plt.legend()
    if output_dir:
        plt.savefig(f"{output_dir}/Tstar_bkt.pdf", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # ---------- 模型3：一级相变 ----------
    x = 1.0 / (L_vals ** d)
    coeffs = np.polyfit(x, T_star_vals, 1)
    a, b = coeffs
    Tc_extrap = b
    results['first'] = {'Tc': Tc_extrap, 'a': a, 'b': b}
    # 绘图
    plt.figure(figsize=(6,5))
    plt.plot(x, T_star_vals, 'o', color='black', label='Data')
    plt.plot(np.linspace(0, max(x), 50), a * np.linspace(0, max(x), 50) + b, 'r--', label=f'1st-order fit, Tc = {Tc_extrap:.4f}')
    if Tc_collapse is not None:
        plt.axhline(y=Tc_collapse, color='blue', linestyle=':', label=f'Tc from collapse = {Tc_collapse:.4f}')
    plt.xlabel(r'$1/L^2$')
    plt.ylabel(r'$T^*(L)$')
    plt.grid(alpha=0.3)
    plt.legend()
    if output_dir:
        plt.savefig(f"{output_dir}/Tstar_first.pdf", bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # 打印结果对比
    print("\n=== T*(L) 外推结果对比 ===")
    for model, res in results.items():
        print(f"{model:8s}: Tc = {res['Tc']:.5f}")
    if Tc_collapse is not None:
        print(f"collapse : Tc = {Tc_collapse:.5f}")

    return results

