import os, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def fit_critical_exponent(temperatures, susceptibilities, Tc_guess=1.4, fit_range=0.5, output_prefix='fit'):
    """
    拟合临界指数 γ 和临界温度 Tc，并绘制诊断图表。
    
    参数:
        temperatures (list/array): 温度序列
        susceptibilities (list/array): 磁化率序列
        Tc_guess (float): Tc 初始猜测值
        fit_range (float): 拟合区间半径
        output_prefix (str): 输出图片文件名前缀
    
    返回:
        dict: 包含拟合参数及统计量的字典，若失败则返回 False
    """
    T = np.array(temperatures, dtype=np.float64)
    X = np.array(susceptibilities, dtype=np.float64)

    # 确保所有值为正（避免对数问题）
    X_positive = X.copy()
    X_min = np.min(X_positive)
    if X_min <= 0:
        X_positive = X_positive - X_min + 0.001

    # 非对称临界幂律
    def critical_law_asymmetric(T, A_above, A_below, Tc, gamma, B_above, B_below):
        result = np.zeros_like(T)
        mask_above = T > Tc
        if np.any(mask_above):
            diff = T[mask_above] - Tc
            diff = np.maximum(diff, 1e-6)
            result[mask_above] = A_above * diff ** (-gamma) + B_above
        mask_below = T < Tc
        if np.any(mask_below):
            diff = Tc - T[mask_below]
            diff = np.maximum(diff, 1e-6)
            result[mask_below] = A_below * diff ** (-gamma) + B_below
        mask_equal = np.abs(T - Tc) < 1e-10
        if np.any(mask_equal):
            result[mask_equal] = 0.5 * (A_above + A_below) * (1e-6) ** (-gamma) + 0.5 * (B_above + B_below)
        return result

    # 选择拟合区间
    mask_above = (T > Tc_guess) & (T < Tc_guess + fit_range)
    mask_below = (T > Tc_guess - fit_range) & (T < Tc_guess)
    mask_combined = (T > Tc_guess - fit_range) & (T < Tc_guess + fit_range)

    n_above = np.sum(mask_above)
    n_below = np.sum(mask_below)

    if n_above < 3 or n_below < 3:
        print(f"警告：一侧数据点不足 (above: {n_above}, below: {n_below})")
        return False

    T_fit = T[mask_combined]
    X_fit = X_positive[mask_combined]
    N_fit = len(T_fit)

    T_above = T[mask_above]
    X_above = X_positive[mask_above]
    T_below = T[mask_below]
    X_below = X_positive[mask_below]

    # 初始猜测
    A_above_guess = (np.max(X_above) - np.min(X_above)) * 0.5
    A_below_guess = (np.max(X_below) - np.min(X_below)) * 0.5
    gamma_guess = 1.0
    B_above_guess = np.min(X_above)
    B_below_guess = np.min(X_below)

    initial_guess = [A_above_guess, A_below_guess, Tc_guess, gamma_guess, B_above_guess, B_below_guess]

    # 参数边界
    bounds = ([0.001, 0.001, Tc_guess - fit_range, 0.1, -np.inf, -np.inf],
              [np.inf, np.inf, Tc_guess + fit_range, 10.0, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(critical_law_asymmetric, T_fit, X_fit,
                               p0=initial_guess, bounds=bounds, maxfev=10000)

        A_above, A_below, Tc_fit, gamma_fit, B_above, B_below = popt
        perr = np.sqrt(np.diag(pcov))
        A_above_err, A_below_err, Tc_err, gamma_err, B_above_err, B_below_err = perr

        fitted = critical_law_asymmetric(T_fit, *popt)
        residuals = X_fit - fitted

        chi2 = np.sum(residuals**2)
        dof = N_fit - 6
        reduced_chi2 = chi2 / dof if dof > 0 else np.nan

        ss_tot = np.sum((X_fit - np.mean(X_fit))**2)
        r2 = 1 - (chi2 / ss_tot) if ss_tot != 0 else 0

        # 图1：主拟合图
        fig1 = plt.figure(figsize=(6, 5))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(T, X, alpha=0.7, s=20, label="Data", color='blue')
        ax1.axvspan(Tc_guess - fit_range, Tc_guess + fit_range, alpha=0.1, color='gray', label='Fit region')
        T_smooth = np.linspace(np.min(T_fit), np.max(T_fit), 200)
        X_smooth = critical_law_asymmetric(T_smooth, *popt)
        ax1.plot(T_smooth, X_smooth, 'r-', linewidth=2, label=f'Fit, γ={gamma_fit:.3f}')
        ax1.axvline(x=Tc_fit, color='green', linestyle='--', alpha=0.7, label=f'Tc = {Tc_fit:.3f}')
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Susceptibility")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        textstr = f'$\\chi^2_\\nu = {reduced_chi2:.3f}$\n(dof = {dof})'
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["RESULTS_DIR"], f"{output_prefix}.pdf"), bbox_inches='tight')
        plt.close(fig1)

        # 图2：T > Tc 对数图
        if len(T_above) > 2:
            fig2 = plt.figure(figsize=(6, 5))
            ax2 = fig2.add_subplot(111)
            dist_above = T_above - Tc_fit
            dist_above = np.maximum(dist_above, 1e-10)
            ax2.loglog(dist_above, X_above, 'ro', alpha=0.7, label='Data (T > Tc)', markersize=6)

            T_line = np.logspace(np.log10(np.min(dist_above)), np.log10(np.max(dist_above)), 100)
            X_line = A_above * T_line ** (-gamma_fit) + B_above
            ax2.loglog(T_line, X_line, 'r-', linewidth=2, label=f'Fit, γ={gamma_fit:.3f}')

            logT = np.log(dist_above)
            logX = np.log(X_above)
            if len(logT) > 1:
                coeffs = np.polyfit(logT, logX, 1)
                line_fit = np.exp(coeffs[1]) * dist_above ** coeffs[0]
                ax2.loglog(dist_above, line_fit, 'r--', alpha=0.7, label=f'Linear fit slope={-coeffs[0]:.3f}')

            ax2.set_xlabel("T - Tc (log scale)")
            ax2.set_ylabel("χ (log scale)")
            ax2.set_title(f"T > Tc   A={A_above:.3f}±{A_above_err:.3f}")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, which='both')
            # y_ticks = [0.001, 0.01, 0.1, 1, 10]
            # ax2.set_yticks(y_ticks)
            # ax2.set_yticklabels(['0.001', '0.01', '0.1', '1', '10'])
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["RESULTS_DIR"], f"{output_prefix}1.pdf"), bbox_inches='tight')
            plt.close(fig2)

        # 图3：T < Tc 对数图
        if len(T_below) > 2:
            fig3 = plt.figure(figsize=(6, 5))
            ax3 = fig3.add_subplot(111)
            dist_below = Tc_fit - T_below
            dist_below = np.maximum(dist_below, 1e-10)
            ax3.loglog(dist_below, X_below, 'bo', alpha=0.7, label='Data (T < Tc)', markersize=6)

            T_line = np.logspace(np.log10(np.min(dist_below)), np.log10(np.max(dist_below)), 100)
            X_line = A_below * T_line ** (-gamma_fit) + B_below
            ax3.loglog(T_line, X_line, 'b-', linewidth=2, label=f'Fit, γ={gamma_fit:.3f}')

            logT = np.log(dist_below)
            logX = np.log(X_below)
            if len(logT) > 1:
                coeffs = np.polyfit(logT, logX, 1)
                line_fit = np.exp(coeffs[1]) * dist_below ** coeffs[0]
                ax3.loglog(dist_below, line_fit, 'b--', alpha=0.7, label=f'Linear fit slope={-coeffs[0]:.3f}')

            ax3.set_xlabel("Tc - T (log scale)")
            ax3.set_ylabel("χ (log scale)")
            ax3.set_title(f"T < Tc   A={A_below:.3f}±{A_below_err:.3f}")
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, which='both')
            # ax3.yaxis.set_major_formatter(ticker.ScalarFormatter())
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["RESULTS_DIR"], f"{output_prefix}2.pdf"), bbox_inches='tight')
            plt.close(fig3)

        # 控制台输出
        print("\n=== Critical exponent fitting results ===")
        print(f"Tc = {Tc_fit:.4f} ± {Tc_err:.4f}")
        print(f"γ  = {gamma_fit:.4f} ± {gamma_err:.4f}")
        print(f"R²  = {r2:.4f}")
        print(f"χ²  = {chi2:.4f}")
        print(f"Reduced χ² = {reduced_chi2:.4f} (dof = {dof})")
        print(f"T > Tc: A+ = {A_above:.4f} ± {A_above_err:.4f}, B+ = {B_above:.4f}")
        print(f"T < Tc: A- = {A_below:.4f} ± {A_below_err:.4f}, B- = {B_below:.4f}")

        return {
            'Tc': Tc_fit, 'Tc_err': Tc_err,
            'gamma': gamma_fit, 'gamma_err': gamma_err,
            'A_above': A_above, 'A_above_err': A_above_err,
            'A_below': A_below, 'A_below_err': A_below_err,
            'B_above': B_above, 'B_below': B_below,
            'r2': r2,
            'chi2': chi2, 'reduced_chi2': reduced_chi2, 'dof': dof
        }

    except Exception as e:
        print(f"拟合失败: {e}")
        return False

# ==================== 第四部分：TWO-NN 本征维数计算与分析 ====================
def plot_fit_subplots_from_fitdata(fit_data_all, output_dir, model_label="", size_label="L"):
    """
    直接利用 fit_data_all 字典绘制每个 L 的 TWO-NN 拟合子图，无需重新计算。
    fit_data_all 格式: { L: [(temp, slope, x_array, y_array), ...] }
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    for L, data_list in fit_data_all.items():
        if not data_list:
            continue
        # data_list 中的每个元素是 (temp, slope, x, y)
        n_total = len(data_list)
        mid = (n_total + 1) // 2
        halves = [data_list[:mid], data_list[mid:]]
        cols = 5
        for part_idx, sublist in enumerate(halves):
            if not sublist:
                continue
            n = len(sublist)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
            if rows == 1 and cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx, (temp, slope, x, y) in enumerate(sublist):
                ax = axes[idx]
                ax.scatter(x, y, s=5, alpha=0.5, color='blue', label='Data')
                x_fit = np.linspace(x.min(), x.max(), 50)
                y_fit = slope * x_fit
                ax.plot(x_fit, y_fit, 'r-', lw=1.5, label=f'Fit: d={slope:.2f}')
                # 计算 R²
                y_pred = slope * x
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
                ax.set_title(f'T={temp:.3f}, d={slope:.2f}, $R^2$={r2:.3f}', fontsize=10)
                ax.set_xlabel('ln(μ)')
                ax.set_ylabel('-ln(1-F)')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            
            for idx in range(n, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            fname = f"2NN_fit_{size_label}{L}_{model_label}_part{part_idx+1}.pdf" if model_label else f"2NN_fit_{size_label}{L}_part{part_idx+1}.pdf"
            save_path = os.path.join(output_dir, fname)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"保存: {save_path}")