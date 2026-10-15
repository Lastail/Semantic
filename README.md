# Semantic

研究大语言模型生成文本在不同采样温度 \(T\) 与文本长度 \(L\) 下的几何结构和物理涨落。项目包含**相互独立的两条分析路线**：

- **几何方法**：representation → 文本距离 → TWO-NN 内禀维度 \(I_d(T,L)\) → FSS。
- **物理方法**：input token embeddings → 位置连通关联 \(C_T(i,j)\) → susceptibility \(\chi(T,L)\)。

两条路线共享生成文本，但 susceptibility **不依赖** static_mean、contextual_mean 或 TWO-NN。

> 本 README 根据目前讨论的架构编写，并未逐文件核验集群上的最新版本。示例文件名与 YAML 字段需与你实际的 `pipeline.py`、`run_susceptibility.py` 对齐。

## 目录

```text
Semantic/
├── configs/
│   ├── default.yaml
│   ├── qwen4b.example.yaml
│   └── experiments/
├── prompts/
├── src/semantic/
│   ├── config.py
│   ├── generation/                 # client.py, sampler.py
│   ├── representations/            # 几何方法
│   │   ├── base.py
│   │   ├── mean_base.py
│   │   ├── static_mean.py
│   │   ├── contextual_mean.py
│   │   ├── positional.py
│   │   └── hidden_positional.py
│   ├── metrics/
│   │   ├── two_nn.py
│   │   ├── correlation.py
│   │   └── susceptibility.py       # 物理方法
│   ├── fss/
│   └── pipeline.py
├── scripts/
│   ├── generate.py
│   ├── run_fss.py
│   ├── run_susceptibility.py
│   └── plot_susceptibility.py      # 如已安装，可独立绘图
├── tests/
├── data/raw/                       # 不上传
├── data/processed/                 # 不上传
├── models/                         # 不上传
├── outputs/                        # 不上传
├── logs/                           # 不上传
├── pyproject.toml
├── .gitignore
└── README.md
```

## 安装与配置

```bash
cd /public/home/ruanhj/Semantic
python -m pip install -e .
mkdir -p logs outputs data/raw
python scripts/run_fss.py -h
pytest -q
```

配置通常按 `default.yaml → model.yaml → experiment.yaml` 合并；如果 `load_config` 使用递归合并，实验 YAML 的：

```yaml
device:
  device: cuda:2
```

将覆盖前面配置的 `device.device`。注意 `CUDA_VISIBLE_DEVICES` 会改变进程内部 GPU 的逻辑编号。公共 `default.yaml` 不应包含真实模型 endpoint、密钥或个人绝对路径。私有 `configs/qwen4b.yaml` 不提交，公开使用脱敏的 `configs/qwen4b.example.yaml`。

## 生成文本

`default.yaml` 的示例：

```yaml
generation:
  max_tokens: 4096
  min_tokens: 4000
  samples_per_temperature: 1000
  max_workers: 70
  thinking_enabled: false
  temperatures:
    start: 0.0
    stop: 3.0
    step: 0.2
```

运行（需要部署好的模型 API，且 generate.py 支持 YAML 温度网格）：

```bash
nohup python -u scripts/generate.py \
  --model-config configs/qwen4b.yaml \
  --prompt prompts/wiki_en.txt \
  --out data/raw/qwen4b_wiki_en_261008.json \
  > logs/qwen4b_generate.log 2>&1 &
```

使用相同输出路径可尝试从 `.checkpoint.jsonl` 续采，不要添加 `--no-resume`。

## 几何路线：TWO-NN 与 FSS

representation 统一返回 `compute_neighbors(texts, length, cache_path=None) -> (r1, r2)`。其中 \(\mu=r_2/r_1\)，再由 TWO-NN 估计 \(I_d(T,L)\)。

| 类型 | token 特征 | 距离 |
|---|---|---|
| `static_mean` | 输入词嵌入 | 位置均值向量的欧氏距离 |
| `contextual_mean` | 指定层 hidden states | 位置均值向量的欧氏距离 |
| `positional` | 输入词嵌入 | 对应位置的平方差求和再开方 |
| `hidden_positional` | 指定层 hidden states | 对应位置的平方差求和再开方 |

Mean 方法：

\[
v_A=\frac1L\sum_{i=1}^L z_i^A,\qquad d(A,B)=\|v_A-v_B\|_2.
\]

Positional 方法：

\[
d_L(A,B)=\sqrt{\sum_{i=1}^L\|z_i^A-z_i^B\|_2^2}.
\]

其中 \(z_i\) 是 input embedding 或指定 Transformer 层 hidden state。`normalize: false` 表示不额外 L2 归一化。

示例 `configs/experiments/qwen4b_fss_hidden_positional.yaml`：

```yaml
experiment:
  name: qwen4b_hidden_layer4
  input: data/raw/qwen4b_wiki_en_261008.json
representation:
  type: hidden_positional
  layer: 4
  batch_size: 4
  text_block_size: 8
  position_block_size: 128
  cache_dtype: float32
device:
  device: cuda:0
fss:
  sizes: [100, 500, 1000, 2000, 3000]
  size_type: token
  mu_cutoff: 0.2
  temperature_range: [0.0, 3.0]
  common_ensemble: true
  initial_guess:
    Tc: 1.2
    nu: 1.0
    xi: 0.0
output:
  dir: outputs/qwen4b_hidden_layer4
```

`initial_guess` 的实际层级需按当前 pipeline 核对：历史版本也曾使用顶层 `initial_guess`。

```bash
nohup python -u scripts/run_fss.py \
  --model-config configs/qwen4b.yaml \
  --config configs/experiments/qwen4b_fss_hidden_positional.yaml \
  > logs/qwen4b_fss_hidden.log 2>&1 &
```

其他几何方法只需更换 YAML 的 `representation.type`、适用参数和独立的 `output.dir`。不要让不同数据、模型、层数或算法版本写入同一实验目录。

## 物理路线：Susceptibility

这是一条**独立**计算路线，直接读取原始文本、tokenizer 和 input embeddings。对于温度 T 下 N 篇长度 L 的文本，定义：

\[
m_i=\frac1N\sum_{a=1}^N e_i^{(a)},
\]

\[
G_T(i,j)=\frac1N\sum_{a=1}^N e_i^{(a)}\cdot e_j^{(a)},
\]

\[
C_T(i,j)=G_T(i,j)-m_i\cdot m_j,
\]

\[
\boxed{\chi(T,L)=\frac1L\sum_{i,j=1}^L C_T(i,j)}.
\]

这里减去的是非连通项，使用 `1/L` 归一化，不对每个 token embedding 额外做单位范数归一化。完整求和可用严格等价的流式方差公式计算，避免创建 L×L 关联矩阵。

建议的 `configs/experiments/qwen4b_susceptibility.yaml`：

```yaml
experiment:
  name: qwen4b_susceptibility_connected
  input: data/raw/qwen4b_wiki_en_261008.json
physics:
  observable: susceptibility
  token_source: input_embedding
  sizes: [100, 500, 1000, 2000, 3000]
  common_ensemble: true
  batch_size: 32
device:
  device: cuda:0
output:
  dir: outputs/qwen4b_susceptibility_connected
```

**注意：只有使用已适配 `physics:` 配置的 `run_susceptibility.py` 才能直接运行这份 YAML。** 如果集群仍是旧版本，请以脚本实际读取的字段为准。

```bash
nohup python -u scripts/run_susceptibility.py \
  --model-config configs/qwen4b.yaml \
  --config configs/experiments/qwen4b_susceptibility.yaml \
  > logs/qwen4b_susceptibility.log 2>&1 &
```

结果应保存为 `susceptibility.json`，记录每个 `(T,L)` 的 `n` 和 `susceptibility`。最终只需一张 `susceptibility_vs_temperature.pdf`：横轴 T，纵轴 \(\chi\)，每个 L 一条曲线。建议计算和绘图分离，不必为修改图形重复加载模型。

两条路线可比较临界温度和有限尺寸特征是否一致；不同 observable 的临界指数不要求相等。

## 监控与排错

```bash
pgrep -af 'scripts/(generate|run_fss|run_susceptibility)\.py'
tail -f logs/qwen4b_susceptibility.log
watch -n 2 nvidia-smi
quota -s
du -sh outputs/* 2>/dev/null | sort -hr | head
```

`nohup: ignoring input` 是正常提示。`Disk quota exceeded` 是配额问题；`Bus error` 可能涉及 memmap、文件系统或底层库，不能仅凭该信息认定 GPU OOM。不要随意删除未完成实验的有效 checkpoint。

## GitHub：只上传源码

仓库应提交源码、测试、文档、公开 prompts 和脱敏 YAML；不提交模型权重、文本数据、结果、日志、缓存、checkpoint、密钥。规则见 `.gitignore`。

### 检查当前仓库

```bash
cd /public/home/ruanhj/Semantic
git status --short
git branch --show-current
git remote -v
```

### 添加代码

```bash
git add README.md .gitignore pyproject.toml src/ scripts/ tests/ prompts/
git add configs/default.yaml configs/experiments/
git add configs/*.example.yaml
git diff --cached --stat
git diff --cached --name-only
```

请检查暂存区是否包含私有地址、真实密钥、大型文件或数据。如果某个目录不存在，跳过对应 `git add`。

### 提交并推送

```bash
python -m compileall -q src scripts
pytest -q
git diff --cached --check
git commit -m "Update geometry and susceptibility analysis"
git push origin main
```

`main` 应替换成 `git branch --show-current` 的真实分支。若还没有 GitHub 远程仓库，先在 GitHub 创建空仓库，再用 `git remote add origin <YOUR_REPO_URL>` 添加（已有 origin 不要重复添加）。

### 已被 Git 跟踪的数据

`.gitignore` 不会自动忽略已经被跟踪的文件。确认后可执行：

```bash
git rm -r --cached --ignore-unmatch data/raw data/processed models outputs logs
git status --short
```

`--cached` 不删除集群磁盘上的文件，但会在新提交中停止跟踪。它也**不会清除旧 Git 历史中的大文件或凭证**。如已泄露密钥应立即轮换，清理历史需要单独计划。

建议每次运行都使用独立输出目录，并在未来给 pipeline 增加实验身份校验和运行锁，避免静默缓存污染。
