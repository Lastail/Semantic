import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
def intrinsic_dimension_two_nn(points, discard_fraction=0, metric='euclidean', n_jobs=-1, return_fit_data=False):
    """使用 TWO-NN 方法计算点集的本征维数"""
    points = np.asarray(points)
    n_samples = points.shape[0]
    if n_samples < 3:
        raise ValueError(f"样本数不足3，当前 {n_samples}")
    nn = NearestNeighbors(n_neighbors=3, metric=metric, n_jobs=n_jobs)
    nn.fit(points)
    dist, _ = nn.kneighbors(points)
    r1 = dist[:,1]
    r2 = dist[:,2]
    eps = 1e-12
    r1 = np.where(r1==0, eps, r1)
    r2 = np.where(r2==0, eps, r2)
    mu = r2 / r1
    mu = mu[np.isfinite(mu)]
    mu = np.maximum(mu, 1.0)
    mu_sorted = np.sort(mu)
    keep = int(len(mu_sorted) * (1 - discard_fraction))
    mu_keep = mu_sorted[:keep]
    N = len(mu_keep)
    j = np.arange(1, N+1)
    F = j / (N+1)
    x = np.log(mu_keep)
    y = -np.log(1 - F)
    slope = np.sum(x*y) / np.sum(x**2)
    y_pred = slope * x
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (len(x)-1)
    std_slope = np.sqrt(mse / np.sum(x**2))
    if return_fit_data:
        return slope, std_slope, x, y
    else:
        return slope

def compute_intrinsic_dimensions_from_vectors(vecs, temps, size, mu_cutoff=0.2):
    """
    从已白化/归一化的向量 (N, d) 与对应温度 (N,) 直接计算 TWO-NN 内在维度。
    与原版 compute_intrinsic_dimensions 的唯一区别：跳过「从文本提取向量」，
    改为「按温度分组取白化向量」。token 版与 block 版通用。
    返回 ({size: {'temps','ids','stds'}}, {size: [(temp, slope, x, y), ...]})
    """
    temps_list, ids_list, stds_list = [], [], []
    fit_data = []

    for t in np.unique(temps):
        mask = temps == t
        group_vecs = vecs[mask]
        n = group_vecs.shape[0]
        if n < 3:                     # 对应原版 len(texts) < 3：TWO-NN 至少需要 3 个样本
            continue

        slope, std_slope, x, y = intrinsic_dimension_two_nn(group_vecs, return_fit_data=True)

        if mu_cutoff:
            mask_x = x <= mu_cutoff
            if mask_x.sum() >= 2:     # 与原版一致：数据点不足 2 个则不重拟合
                x, y = x[mask_x], y[mask_x]
                slope = np.sum(x * y) / np.sum(x**2)
                y_pred = slope * x
                mse = np.sum((y - y_pred)**2) / (len(x) - 1)
                std_slope = np.sqrt(mse / np.sum(x**2))

        temps_list.append(t)
        ids_list.append(slope)
        stds_list.append(std_slope)
        fit_data.append((t, slope, x.copy(), y.copy()))

    outcomes, fit_data_all = {}, {}
    if temps_list:
        idx = np.argsort(temps_list)
        outcomes[size] = {
            'temps': [temps_list[i] for i in idx],
            'ids': [ids_list[i] for i in idx],
            'stds': [stds_list[i] for i in idx],
        }
        fit_data.sort(key=lambda e: e[0])
        fit_data_all[size] = fit_data

    return outcomes, fit_data_all
