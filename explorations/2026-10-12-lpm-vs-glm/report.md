# LPM BCD 与 GLM（IRLS / Newton）：offset、速度与可替代性

日期：2026-10-12 · 状态：探索结论，未实施 · 涉及：[`linear_product_model`](../../src/quantbullet/linear_product_model/)、
[turnover baseline](../../projects/freddie_prepayment/design/turnover.md) · 实施设计：[design.md](design.md)

## 问题

1. `statsmodels` GLM 的 `offset` 如何工作（以 Binomial 为例），能否用它冻结已有模型、只拟合新增部分。
2. IRLS 速度如何；scikit-learn 是否有带 offset 的等价功能。
3. `LinearProductRegressorBCD`（下称 LPM）与 GLM 相比是否多余，能否被替代。

## 结论摘要

- offset 是线性预测器中系数固定为 1 的项；Binomial/logit 下必须是 log-odds。用它冻结旧模型可行，
  但估计的是"冻结部分之上的增量"，冻结部分有偏时新系数也会偏。
- statsmodels GLM 结果正确但慢：1M 行 × 87 参数需 29–37s，是 sklearn `newton-cholesky` 的 8–10 倍。
  只拟合少量参数的 offset 场景可用（8 个参数，1M 行 1.3s）。
- sklearn 1.9.1 的 GLM 没有 offset 参数。Poisson + log link 可用 `y / e^offset` 加 `sample_weight = e^offset`
  精确等价，实测系数差 ≤ 4e-6、快 5–7 倍；Logistic 没有精确绕法。
- LPM 不是 GLM，statsmodels 不能直接拟合；但同样基函数的 log-link GLM 也是乘积因子模型。
  模拟中两者 holdout log loss 差距 ≤ 1.3e-5（约 0.02%）。
- 特征独立时 LPM 与 sklearn Newton 一样快；相关系数 0.9 时 BCD 需要 26–52 sweeps，慢 3–6 倍，
  且默认 `ftol=1e-5` 会在 loss 仍下降时停止。Newton 迭代数不随相关性变化。
- 判断：LPM 的求解器层基本可被 log-link joint Newton 替代；接口、submodel、interaction、toolkit 和 diagnostics 层不多余。
- 替代的求解器应是 sklearn `newton-cholesky`、glum 或自研 Newton，不是 statsmodels（比现有 BCD 慢 8–10 倍）。
  statsmodels 适合少量参数的 offset 拟合和推断检查。
- Turnover 全量实测（17.0M 行，in-sample，第 9 节）：log-link GLM 与 LPM 拟合质量相同（deviance 低 0.0045%，
  各 facet A/E 误差几乎相同）；分块 dense Newton 7 步收敛，227s 对 LPM 371s，快约 1.6 倍，代价是多约 10.6 GB 内存。
  LTV、factor、HPI ratio 四条相互依赖的曲线在两模型间相差最多 20%，说明它们本身识别较弱。时间外验证尚未在全量上运行。

## 1. GLM offset 的机制

```text
g(mu_i) = eta_i = x_i' beta + offset_i
Binomial（默认 logit）：log(p_i / (1 - p_i)) = offset_i + x_i' beta
```

IRLS 每轮：

1. `eta = X beta + offset`，`mu = logistic(eta)`，`w = mu (1 - mu)`。
2. Working response `z = eta + (y - mu) / w - offset`。
3. 以 `w` 加权，`z` 对 `X` 做 WLS，更新 `beta`。

offset 影响 `mu` 和权重，但不进入 `beta`。

验证见 [glm_offset_demo.py](glm_offset_demo.py)：n=50,000，真实模型 `-2 + 0.8 x1 + 0.5 x2`，x2 与 x1 相关；
冻结 `-2 + 0.8 x1`，只拟合 `const + x2`。

| 拟合 | const | x1 | x2 |
|---|---|---|---|
| statsmodels offset | -0.002 | 固定 0.8 | 0.4976 |
| 手写 IRLS | -0.002 | — | 0.4976 |
| 联合拟合 | -2.0021 | 0.8004 | 0.4975 |
| offset = 只用 x1 拟合的 base | -0.176 | 固定 1.010 | 0.447 |

注意事项：

- offset 在 link 尺度：Binomial/logit 传 log-odds，不传概率。
- 冻结部分有偏时新系数随之偏。上表最后一行中，只用 x1 拟合的 base 让 x1 系数（1.01）吸收了相关的 x2 效应，
  x2 降为 0.447。offset 拟合是"约束部分系数后的 MLE"，不是联合重新拟合。
- `predict(exog=new)` 不传 `offset` 时**静默按 0 处理**，没有 warning（实测 0.47，正确值 0.118）。
- `exposure` 等于把 `log(exposure)` 加进 offset，只允许 log link；Binomial/logit 下直接 `ValueError`。

## 2. 速度

### 设置

[bench_speed.py](bench_speed.py)：4 个连续变量各 7 个 flat-ramp 基函数，加 10 水平与 50 水平类别变量；
Bernoulli 事件率约 1.5%，按 LPM 乘积形式生成（对 LPM 有利）；80/20 holdout。
GLM 设计矩阵 87 列（单一截距，类别去掉首水平）；LPM 每个连续块自带常数列，类别块为完整 one-hot。

环境：Windows 11，28 logical CPUs，64 GB；statsmodels 0.15.0，scikit-learn 1.9.1，NumPy 2.5.3，SciPy 1.16.3。
计时为单次运行。

| 方法 | 1M fit | 4M fit | holdout log loss 与最优之差（1M / 4M） |
|---|---|---|---|
| statsmodels GLM Poisson/log | 29.1s（8 iter） | 122.0s | +8.8e-6 / +7.5e-6 |
| statsmodels GLM Binomial/logit | 37.3s（9 iter） | 154.6s | +10.2e-6 / +6.0e-6 |
| sklearn PoissonRegressor `newton-cholesky` | 3.8s（7 iter） | 15.5s | +12.6e-6 / +9.2e-6 |
| sklearn LogisticRegression `newton-cholesky` | 3.8s（6 iter） | 14.0s | +12.1e-6 / +6.5e-6 |
| sklearn LogisticRegression `lbfgs` | 3.5s（38 iter） | 13.5s | +61.5e-6 / +21.3e-6 |
| LPM BCD `loss="poisson"` | 3.4s（6 sweeps） | 12.0s | +1.2e-6 / 0 |
| LPM BCD `loss="mse"` | 3.4s（6 sweeps） | 15.6s | 0 / +5.6e-6 |

Holdout log loss 基数约 0.070（1M）和 0.071（4M）。

Offset-only 拟合：冻结 LPM 的其他块，`offset = log(leave-out prediction)`，只拟合一个连续块的 8 个参数。

| 方法 | 1M | 4M |
|---|---|---|
| statsmodels GLM Poisson + offset | 1.28s | 10.33s |
| sklearn Poisson weight trick（第 3 节） | 0.26s | 1.44s |

### 解读

- 所有方法耗时对 n 线性。IRLS 即 Fisher scoring 的 Newton 法：每轮一次 WLS，成本约 n·p²，6–9 轮收敛。
- statsmodels 慢在实现：IRLS 每步默认 `wls_method="lstsq"`，收敛后还用 `pinv` 再做一次完整 WLS 生成结果；
  sklearn `newton-cholesky` 用 X'WX 加 Cholesky。
- `lbfgs` 用时相近但未充分收敛，log loss 明显较差。

## 3. sklearn 与 offset

sklearn 1.9.1 中 `LogisticRegression.fit` 与 `PoissonRegressor.fit` 的签名都只有 `(X, y, sample_weight)`。

Poisson + log link 有精确等价：

```python
e = np.exp(offset)                     # 已有权重 w 时：sample_weight = w * e
PoissonRegressor(alpha=0, solver="newton-cholesky").fit(X, y / e, sample_weight=e)
```

依据是 unit deviance 满足 `d(y, e·m) = e · d(y/e, m)`。实测与 statsmodels offset 结果系数差 ≤ 4e-6。
`alpha` 默认为 1（L2 惩罚），必须显式设为 0。

Logistic 没有精确等价。其他支持 offset 的选项（均未在本环境测试）：

- glum：offset、L1/L2、Poisson/Binomial/Tweedie，高基数类别通过 tabmat 处理；项目未安装。
- LightGBM `init_score`、XGBoost `base_margin`：在冻结模型之上做 boosting。

## 4. LPM 与 log-link GLM 的结构关系

```text
LPM:           mu = s × prod_k (X_k beta_k)            因子在乘数尺度线性，整体多重线性
log-link GLM:  mu = exp(sum_k X_k beta_k) = prod_k exp(X_k beta_k)
```

- LPM 不是 GLM，statsmodels 无法一次拟合。BCD 固定其他块后，每一步恰好是一个 identity-link GLM
  （设计矩阵按其他块的乘积缩放）：[`_solve_block_poisson`](../../src/quantbullet/linear_product_model/regressor.py)
  是 identity-link Poisson 的一步 IRLS，`_solve_one_hot_poisson` 是它的 one-hot 分组求和版本。
- 类别块：两者模型空间相同，只是参数化不同。
- 连续 ramp 块：LPM 在乘数上分段线性，GLM 在 log 乘数上分段线性；使用 flat ramp 时两者在 knots 外都平，
  差异只在 knots 之间的曲线形状。
- 第 2 节数据按 LPM 形式生成，所有方法 holdout log loss 差距 ≤ 1.3e-5（约 0.02%），实际不可区分。

## 5. 特征相关时的收敛

[bench_correlation.py](bench_correlation.py)：同第 2 节设置，4 个连续变量通过共同因子相关，1M 训练行，
BCD 设 `early_stopping_rounds=None`。

| corr(x0, x1) | sklearn Newton（Poisson / Logistic） | LPM `ftol=1e-5` | LPM `ftol=1e-9` | 达到最优 loss 1e-6 以内的 sweep |
|---|---|---|---|---|
| 0.00 | 3.8s / 3.8s（7 / 6 iter） | 2.7s，5 sweeps | 3.6s，7 sweeps | 2 |
| 0.58 | 4.0s / 3.6s | 5.2s，11 sweeps | 7.4s，16 sweeps | 8 |
| 0.89 | 3.9s / 3.5s | 11.6s，26 sweeps | 22.1s，52 sweeps | 29 |

- Newton 迭代数不随相关性变化；BCD 是坐标下降，相关块之间来回调整，sweeps 随相关性增加。
- 相关系数 0.89 时，`ftol=1e-5` 在第 26 sweep 停止，此时 loss 仍在下降（最优约在第 29 sweep）。
  本次对 holdout log loss 的影响 < 1e-6。
- Turnover 模型同时包含 original LTV、factor、HPI ratio 和 updated LTV，且
  `updated LTV = original LTV × factor / HPI ratio` 在 log 空间严格线性相关，属于这类高相关场景；
  turnover 默认 `ftol=1e-5`。

## 6. 哪些可替代，哪些不可替代

| 方面 | LPM（BCD） | log-link GLM（joint Newton） |
|---|---|---|
| 优化问题 | 非凸；尺度不可识别，靠每步归一化和 `global_scalar_` 处理 | Poisson/log、Binomial/logit 下凸，最优唯一 |
| 迭代数 | 随块间相关性增加 | 基本恒定（6–9） |
| 预测正值 | 不保证；solver 用 1e-10 floor，turnover report 需计数负预测 | 天然为正（logit 还保证 < 1） |
| 标准误 | `calculate_feature_group_se` 用 OLS 公式且固定其他块（条件 SE）；Poisson loss 下不适用，忽略块间协方差；classifier 未实现 | Fisher information 给出联合 SE（权重语义见第 7 节） |
| offset / 正则化 | `offset_y` 是响应上的加法平移，不是 link offset；无正则化 | 原生支持 |
| 高基数类别 | one-hot 块用 bincount，O(n) | dense one-hot 每列都计入 n·p²；glum/tabmat 可处理 |
| 因子曲线读法 | 每块归一到均值 1 的乘数加 global scalar | `exp(X_k beta_k)` 后归一化，需自行实现 |
| Frozen submodel | `submodels`，任意 `.predict()` | offset 加入 `log(submodel prediction)`，等价 |
| 分类别曲线 | `InteractionCoef` 及 level 转移 | ramp × one-hot 交互列，log 空间等价 |
| 诊断与报告 | toolkit、implied actual、`MortgageDiagnostics`、ComponentManager | 无，需适配 |

求解器层（上表前五行）GLM 更好或相同；建模和诊断层（后五行）是 LPM 的实际价值，GLM 库不提供，
但都有对应的实现方式。

## 7. 实施前需要决定的问题

候选方案：

- **A. 保持 BCD，修局部问题**：收敛判据（相关块下过早停止）；SE 改为基于全模型联合 Fisher information，或明确标注为条件 SE。
- **B. 可插拔求解器**：保留 LPM 接口、feature groups、submodel、interaction 和 toolkit，新增 log-link joint Newton 后端，
  各块因子取 `exp(X_k beta_k)` 归一化后对外暴露；BCD 保留为可选后端。
- **C. 外部库加薄适配层**：用 glum 或 sklearn 求解，LPM 降为结果容器和诊断层。

待决问题：

1. 业务上是否需要"乘数尺度线性"的因子曲线（例如与既有模型可比、knots 间线性插值）？若不需要，log-link 的形状差异可接受。
2. Toolkit 依赖的 model 接口：`coef_`（含 `InteractionCoef`）、`global_scalar_`、`feature_groups_`、`submodels_`、
   `offset_y`、`loss_`、`fit_metadata_`、`predict`、`leave_out_feature_group_predict`、`single_feature_group_predict`，
   以及收敛图用的 `loss_history_`、`best_loss_`、`global_scalar_history_`、`global_scalar_step_history_`。新后端需逐一满足或适配。
   Poisson implied actual 公式（`p` 为当前 component，`m` 为其他乘积）用归一化后的 `exp(X_k beta_k)` 应仍可计算，需验证。
3. 权重语义：turnover 使用 balance weight。sklearn `sample_weight`、statsmodels `freq_weights` 与 `var_weights`
   点估计相同但 SE 不同，需要确定哪种对应余额加权。
4. 内存与 dtype：turnover 使用 float32 block arrays；sklearn `newton-cholesky` 是否保持 float32，
   以及 state、month 等 dense one-hot 的成本，需要实测。
5. 高基数类别和 interaction 是否需要专用实现：glum/tabmat，或自研 Newton 中对 one-hot 块用分组求和构造 Hessian 分块。
6. 验证方式：在 turnover frame 上用同一 cohort、knots、weights 拟合两种模型，比较 weighted Poisson deviance、
   A/E facets、因子曲线、fit 时间和负预测个数；再做时间外样本验证。In-sample 部分已完成（第 9 节），时间外待跑。
7. LTV / factor / HPI ratio 曲线弱识别（第 9 节）：无论选哪种模型，解读单条曲线或在 turnover 之上用 offset
   叠加新组件前，是否需要约束、合并或换参数化（例如只保留其中三个）？

## 8. 实验局限

- 合成数据：87 个参数，无观测权重、无 interaction，类别变量相互独立。真实 turnover 有更多块、balance weights、
  age×purpose interaction 和 float32 存储。
- 只测了 `LinearProductRegressorBCD`；`LinearProductClassifier*`（clip 后 log loss + L-BFGS）未测。
- 只比较 Poisson/log 与 Binomial/logit；未测 cloglog 和 glum。
- 计时为单次运行，BCD 结果受 `ftol`、`early_stopping_rounds` 影响。

## 9. Turnover 实测对比

[compare_turnover.py](compare_turnover.py) 读取 turnover config 和 `turnover_frame.parquet`，复用 `fit_turnover` 的
`to_model_data`、`build_toolkit`、`make_container`：cohort、clip、knots、categories、age×purpose interaction 和
balance weights 与 fit stage 一致。

- **LPM**：`LinearProductRegressorBCD(loss="poisson")`，参数取自 config（`--lpm-ftol` 可覆盖 ftol）。
- **log-link Poisson GLM**：同一组 blocks。FlatRamp 去掉 bias 列（全局一个 intercept），类别 block 去掉余额最大的 level，
  age interaction 为 ramps × purpose one-hot，所有列标准化，共 156 列。
  用分块 Newton/IRLS 求解：design 以 float32 缓存（全量约 10.6 GB，`--stream-design` 改为每轮重建），
  Hessian 用 float32，gradient 和 deviance 用 float64，带 step halving。
  300k 样本上与 sklearn `newton-cholesky` 对照：deviance 差 5e-12，线性预测器最大差 2e-6。
- LPM 可能给出负预测，此时严格 Poisson deviance 无定义；两模型都另报 floor 到 1e-10 的 deviance（与 LPM solver 一致）。
- 抽样运行中常有无事件的 level（如 VI、GU、PR），其 MLE 为 −∞，Newton 会很慢，需加 `--glm-l2 1e-8`。
  全量数据每个 level 至少有 4 个事件，默认 `--glm-l2 0`。

输出写到 `<output_root>/lpm_vs_glm/<run>/`，不进 repo：

| 文件 | 内容 |
|---|---|
| `summary.json` | 行数、拟合参数、迭代与收敛、耗时、两模型的 `prediction_metrics` 与 floor deviance |
| `ae_bins.csv`、`ae_summary.csv` | 各 facet 的余额加权 A/E：6 个类别、reporting month/year、数值 fit 输入（report 的 bin 宽度）；summary 为 ≥ 500 rows 的 bins 上 \|A/E − 1\| 的余额加权 RMS 与最大值 |
| `factor_curves.csv`、`categorical_factors.csv`、`factors.pdf` | 两模型因子，均归一到余额加权均值 1；interaction 曲线在每个 purpose 内归一，其均值水平并入 purpose 因子 |

### 全量 in-sample 结果（2026-10-12）

17,042,421 行，config 默认参数（LPM `ftol=1e-5`、最多 60 sweeps），GLM `l2=0`。同一台机器（第 2 节环境），单次运行。

| | LPM BCD | log-link GLM（分块 Newton） |
|---|---|---|
| Fit 时间 | 371s | 227s（design 31s + Newton 196s） |
| 迭代 | 16 sweeps，`ftol` 停止；最优在最后一轮，loss 仍在下降 | 7 步（8 passes），第 5 步与最终值的相对差 8e-8 |
| 余额加权 Poisson deviance | 0.04097962 | 0.04097777（−1.85e-6，−0.0045%） |
| 等权 Poisson deviance | 0.0464665 | 0.0464164 |
| 整体 A/E | 1.0000 | 1.0000 |
| 负预测 | 0 | 0 |
| 预测范围 | 0.00018–0.101 | 0.00020–0.117 |
| 额外内存 | — | float32 design 缓存约 10.6 GB |

A/E（≥ 500 rows 的 bins，余额加权 RMS of \|A/E − 1\|）：

| Facet | LPM | GLM |
|---|---|---|
| 6 个类别变量 | ≤ 0.0002（最大 0.0012） | 0（score equations 精确满足） |
| reporting month / year | 0.1110 / 0.0818 | 0.1117 / 0.0824 |
| age | 0.0491 | 0.0486 |
| incentive | 0.0142 | 0.0149 |
| updated LTV | 0.0221 | 0.0217 |
| factor | 0.0263 | 0.0262 |
| 其他数值输入 | 0.012–0.022 | 0.012–0.022 |

因子比较（数据 1–99% 分位范围内，\|GLM / LPM − 1\| 的最大值）：

- incentive、SATO、original FICO、real original balance：≤ 0.7%。
- age × purpose：C 5.7%、N 12.2%、P 4.1%，主要是曲线低值段和整体水平的差异。
- original LTV 12.5%、updated LTV 18.5%、factor 20.3%、HPI ratio 7.2%（全范围 87%，在 0.8–0.97 的稀疏段）。
- 类别因子（≥ 0.1% 余额的 levels）：≤ 0.5%。

解读：

1. **拟合质量相同。** In-sample 只有 0.0045% 的 deviance 差距，各 facet A/E 也几乎相同，乘数尺度线性与 log 尺度线性的形状差异在这个数据上不重要。
2. **Newton 更快，但差距不大。** 1.6 倍来自迭代数（7 对 16），每次 pass 约 25s，是 BCD 每个 sweep（约 23s）的同量级：
   156 列中 74 列是 one-hot，dense Hessian 没有利用这一点。用分组求和构造 one-hot 块、放宽 tol（第 5 步已足够）都可以继续缩短。
3. **LTV / factor / HPI 曲线弱识别。** 四者满足 `updated LTV = original LTV × factor / HPI ratio`，两个模型总预测几乎相同，
   但这四条曲线相差最多 20%：差异来自两种参数化分配联合效应的方式，不是拟合优劣。
   [turnover design](../../projects/freddie_prepayment/design/turnover.md) 已提示要同时看四条曲线；这对解读和在 turnover 之上用 offset 叠加 refi 组件都有影响。
4. **时间外表现尚未在全量上验证。** 抽样 holdout（最后 12 个月）结果方向不一致：300k 样本 GLM test deviance 低 0.7%，
   200k 样本反而高 4.8%。小样本中很多 level 在训练期没有或只有极少事件，结果被噪声主导，需要用 `--holdout-months 12` 全量运行。

## 10. GLM 版 turnover report

[fit_turnover_glm.py](fit_turnover_glm.py) 用与 `fit_turnover` 相同的数据拟合 log-link GLM，再把它精确写成
`LinearProductRegressorBCD` 的形式，使 `report_turnover` 不经修改即可生成同样结构的 PDF：

- 数值 block：frozen submodel，`predict(X_k) = exp(X_k @ a_k + b_k)`，归一到余额加权均值 1。
  用 sklearn `PoissonRegressor` 手动设定系数实现，pickle 中只有 quantbullet / sklearn 的标准类。
- age × purpose：`InteractionCoef`，每个 purpose 一个上述 submodel；类别 block：每个 level 的因子向量。
- `global_scalar_` 吸收 intercept 和所有均值；保存前校验 `model.predict()` 与 GLM 直接预测的相对差（全量为 0 量级，阈值 1e-6）。
- Convergence 页显示 Newton 每步的 deviance 和对应的 global scalar；Fit summary 的 Loss 字段标为
  `poisson / log-link GLM`，Best iteration 为 Newton 步数。

产物写到与 turnover output 并列的 `<output_root>_glm/`：`turnover_model.pkl`、`turnover_predictions.parquet`、
`turnover_frame.parquet` 的 hard link（不同卷时复制）、供 report 使用的 `turnover_glm.toml`，以及 report 生成的 `turnover_report.pdf`。
原 turnover 产物不变。

## 复现

从 repo root 运行（Windows 用 `.venv\Scripts\python.exe`）：

```bash
.venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/glm_offset_demo.py
.venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/bench_speed.py 1000000
.venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/bench_correlation.py 1000000
```

`bench_speed.py` 的参数为训练行数。4M 行约 6–7 分钟，dense 设计矩阵（5M × 87，float64）约 3.5 GB，statsmodels 另有多份副本。

Turnover 对比需要 `.env` 中的 `FREDDIE_DATA_ROOT` 和已生成的 `turnover_frame.parquet`：

```bash
.venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/compare_turnover.py --config projects/freddie_prepayment/turnover.example.toml
```

常用选项：`--holdout-months 12`（最后 12 个 reporting months 作时间外样本）；
`--sample-rows 300000 --glm-l2 1e-8 --sklearn-check`（快速检查，同时用 sklearn 验证 Newton 解）；
`--lpm-ftol 1e-7`（LPM 收紧收敛判据后能再改进多少）。

GLM 版 turnover report（第一步打印第二步需要的 config 路径，位于 `<output_root>_glm/`）：

```bash
.venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/fit_turnover_glm.py --config projects/freddie_prepayment/turnover.example.toml
.venv/Scripts/python.exe -m projects.freddie_prepayment.report_turnover --config <output_root>_glm/turnover_glm.toml
```

抽样检查用 `--smoke-rows 200000 --glm-l2 1e-8 --output-root <另一个仓库外目录>`。
