import os, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy.polynomial import Chebyshev
def perform_data_collapse(outcomes, output_fig_path, guess_Tc=1.4, guess_nu=1.0, guess_xi=0.0,
                         poly_degrees=[5, 6, 7, 8], L_subsets=None):
    L_vals = sorted(outcomes.keys())
    
    # 保存原始数据用于绘图
    temps_list = []
    ids_list = []
    for L in L_vals:
        d = outcomes[L]
        temps = np.array(d['temps'])
        ids_arr = np.array(d['ids'])
        idx = np.argsort(temps)
        temps_list.append(temps[idx])
        ids_list.append(ids_arr[idx])
    
    def fit_polynomial_chebyshev(x_data, y_data, degree):
        """使用切比雪夫多项式拟合数据（数值最稳定）
        
        返回:
            p_fit: 拟合函数（可调用）
            y_fit: 拟合值数组
            residual: 相对残差
        """
        # 只使用有限值
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        # 检查数据点是否足够
        if len(x_clean) < degree + 1:
            return None, None, 1e9
        
        # 切比雪夫拟合（自动处理数值稳定性）
        # 注意：Chebyshev.fit 内部会进行区间映射到 [-1, 1]
        coefs = Chebyshev.fit(x_clean, y_clean, degree)
        
        # 返回可调用的拟合函数
        def p_fit(x):
            return coefs(x)
        
        # 计算拟合值
        y_fit = p_fit(x_clean)
        
        # 计算相对残差（论文公式 C1）
        residual = np.sqrt(np.sum((y_clean - y_fit)**2)) / np.sqrt(np.sum(y_clean**2))
        
        return p_fit, y_fit, residual
    
    def compute_and_optimize_residuals(params, poly_degree, L_subset):
        """融合版本：计算给定参数下的残差（可直接用于优化）
        
        参数:
            params: [Tc, nu, xi] 或 [Tc, a, xi] (BKT情况)
            poly_degree: 多项式次数
            L_subset: 系统尺寸子集
        
        返回:
            residual: 相对残差（越小越好）
        """
        Tc, nu, xi = params
        
        # 物理约束检查
        if nu <= 0 or Tc <= 0:
            return 1e9
        
        # 收集当前L子集的所有缩放数据
        all_x = []
        all_y = []
        
        for L in L_subset:
            d = outcomes[L]
            temps = np.array(d['temps'])
            ids_arr = np.array(d['ids'])
            
            # 标度变换
            t = (temps - Tc) / Tc
            x = L ** (1.0 / nu) * t
            y_scaled = ids_arr / (L ** xi)
            
            # 过滤无效值
            mask = np.isfinite(x) & np.isfinite(y_scaled)
            all_x.extend(x[mask])
            all_y.extend(y_scaled[mask])
        
        if len(all_x) == 0:
            return 1e9
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        
        # 按x排序（便于后续可视化，但不必须）
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]
        
        # 检查数据点是否足够
        if len(all_x) < poly_degree + 1:
            return 1e9
        
        # 切比雪夫多项式拟合
        _, _, residual = fit_polynomial_chebyshev(all_x, all_y, poly_degree)
        
        return residual if residual is not None else 1e9
    
    # 存储所有优化结果
    all_results = []
    
    for poly_degree in poly_degrees: 
        print("degree=", poly_degree)
        # Nelder-Mead 不需要边界，但可能需要更多迭代
        res = minimize(
            lambda params: compute_and_optimize_residuals(params, poly_degree, L_vals),
            [guess_Tc, guess_nu, guess_xi],
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.success:
            Tc_opt, nu_opt, xi_opt = res.x
            residual = res.fun
            all_results.append({
                'Tc': Tc_opt,
                'nu': nu_opt,
                'xi': xi_opt,
                'residual': residual,
                'poly_degree': poly_degree
            })
            print(f"  成功: Tc={Tc_opt:.4f}, nu={nu_opt:.4f}, xi={xi_opt:.4f}, ε={residual:.6f}")
        else:
            print(f"  优化失败: {res.message}")
    
    # 计算统计量（平均值和标准差）
    Tc_values = [r['Tc'] for r in all_results]
    nu_values = [r['nu'] for r in all_results]
    xi_values = [r['xi'] for r in all_results]
    residuals = [r['residual'] for r in all_results]
    
    Tc_mean = np.mean(Tc_values)
    Tc_std = np.std(Tc_values)
    nu_mean = np.mean(nu_values)
    nu_std = np.std(nu_values)
    xi_mean = np.mean(xi_values)
    xi_std = np.std(xi_values)
    
    # 找到最佳结果（残差最小）
    best_idx = np.argmin(residuals)
    best_result = all_results[best_idx]
    
    # 使用最佳参数绘制坍塌图
    Tc_best = best_result['Tc']
    nu_best = best_result['nu']
    xi_best = best_result['xi']
    
    # 计算所有缩放数据用于绘图
    all_x_plot = []
    all_y_plot = []
    for L in L_vals:
        d = outcomes[L]
        temps = np.array(d['temps'])
        ids_arr = np.array(d['ids'])
        
        t = (temps - Tc_best) / Tc_best
        x = L ** (1.0 / nu_best) * t
        y_scaled = ids_arr / (L ** xi_best)
        
        mask = np.isfinite(x) & np.isfinite(y_scaled)
        all_x_plot.extend(x[mask])
        all_y_plot.extend(y_scaled[mask])
    
    all_x_plot = np.array(all_x_plot)
    all_y_plot = np.array(all_y_plot)
    sort_idx = np.argsort(all_x_plot)
    all_x_plot = all_x_plot[sort_idx]
    all_y_plot = all_y_plot[sort_idx]
    
    # 用最佳多项式次数拟合最佳曲线
    best_poly_degree = best_result['poly_degree']
    p_best, y_fit, _ = fit_polynomial_chebyshev(all_x_plot, all_y_plot, best_poly_degree)
    
    plt.figure(figsize=(10, 8))

    colors = ['#0000FF', '#FF7F00', '#8B00FF', '#00FF00', '#FF0000', '#4B0082']
    
    for L, temps, ids_arr, color in zip(L_vals, temps_list, ids_list, colors):
        t = (temps - Tc_best) / Tc_best
        x = L ** (1.0 / nu_best) * t
        y_scaled = ids_arr / (L ** xi_best)
        
        mask = np.isfinite(x) & np.isfinite(y_scaled)
        x_plot = x[mask]
        y_plot = y_scaled[mask]
        
        if len(x_plot) > 0:
            idx = np.argsort(x_plot)
            plt.plot(x_plot[idx], y_plot[idx], 'o-', markersize=4, linewidth=1.5,
                    alpha=0.7, label=f'L={L}', color=color)
    
    plt.xlabel(r'$x = L^{1/\nu} (T-T_c)/T_c$')
    plt.ylabel(r'$I_d / L^{\xi}$')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    plt.axhline(y=0, color='gray', linestyle=':', )
    plt.axvline(x=0, color='gray', linestyle=':', )
    
    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'Tc': Tc_mean,
        'Tc_std': Tc_std,
        'nu': nu_mean,
        'nu_std': nu_std,
        'xi': xi_mean,
        'xi_std': xi_std,
        'Tc_best': Tc_best,
        'nu_best': nu_best,
        'xi_best': xi_best,
        'all_results': all_results,
        'residuals': residuals,
        'best_poly_degree': best_poly_degree
    }

