# Turnover baseline：cohort、model 与 diagnostics

运行方式见 [manual](../README.md)。三个阶段独立，report 仅重读保存的 model、frame 和 predictions。

## 1. Cohort 与 target

以 reporting month t 为单位，从 generic panel 选择连续上月 `f_pre_status=CURRENT`、previous balance >0、age >=1，
排除当前及之前已观察到 modification 的 rows，以及 post-exit、未知 exit、缺失 exit month、event month mismatch 和无原因零余额。
不根据当月是否 CURRENT 筛选，CURRENT→delinquency 仍是 negative observation。

项目 target `y_full_prepay=1`：`zero_balance_code=01`、event month 等于 reporting month，且早于披露 maturity。
其他已知结果为 0；code 01 但 maturity 缺失时排除。所有规则只作用于 modeling frame，generic panel 不修改。

Fitting cohort 为 `c_incentive <= -0.5`，数值 features 要求非空且有限。负 incentive 减少普通 rate-refinancing 的影响，
但不能从 payoff reason 识别卖房/cash refinance，因此这个模型是 prepayment baseline，不是已观测的纯 housing turnover。
样本主要来自 2022 年以后。本阶段全量样本内拟合，不划分 train/test。

`weight = c_prev_balance / mean(c_prev_balance)`：paid-off row 不因 current balance=0 失去权重。
余额加权月度 mean 是 full-payoff SMM proxy，不包括 partial curtailment，也未修正当月 scheduled amortization。

## 2. Multiplicative model

```text
pred_turnover = global_scalar × product(numeric blocks) × product(categorical blocks)
```

使用 `LinearProductRegressorBCD(loss="poisson")`。`c_age_fit` 按 `f_purpose` 各估一条 ramp，其余 block 是 main effects。
每个 block 在 fit 中归一化，global scalar 表示整体 response level。
默认最多 60 sweeps、10-round early stopping、ftol=1e-5，不缓存 QR。ftol 比较最近 5 个 sweep 的相对 loss 改进，低于 ftol 即停止；1e-8 实际从不触发。Early stopping 在当前 loss 不低于 10 个 sweep 前时停止。
完整 one-hot 的普通 Poisson blocks 使用 category codes + grouped sums 更新，保留原 IRLS epsilon-floor 和 ridge=1e-8；
其他 design matrices、numeric / interaction curves 和 MSE 路径仍使用原 solver。Codes 仅在当前 fit 中缓存，不写入 artifacts。
`model.fit_timing_` 记录 setup、各 block 累计时间、每个 sweep 时间和总 fit 时间。
Metadata 的 `timing_seconds` 分别记录 read_frame、model_data、toolkit、container、fit、predict、metrics 和 metadata，
`actual_sweeps` 是实际执行轮数；`fit_seconds` 仍只计 `model.fit()`。Artifact write 和全流程总时间完成后打印，不回写 pickle。

Prepared frame 保留 raw values。Fit 按 [fit](../fit_turnover.py) 的 `CLIP` 生成 `_fit` inputs：age、incentive、SATO、original FICO、original LTV、updated LTV、real original balance、balance factor 和 HPI ratio。改 clip 或 knots 不重写 frame。
同一文件的 `KNOTS` 给出对应 FlatRamp knots。Implied-actual bin widths 在 [report](../report_turnover.py) 的 `IMPLIED_BIN_CONFIG`。
Updated LTV 是 first-lien estimate；`c_hpi_ratio = c_hpi_lag1 / c_orig_hpi`，表示 origination→lag1 的 ZHVI 倍数，不年化，也不是 trailing 24-month HPA。
`1.00` 表示持平，`1.10` 表示累计上涨 10%；fit clip 为 `[0.8, 2.0]`，knots 为 `0.95, 1.0, 1.1, 1.25, 1.5, 1.75`，report bin width 为 `0.1`。
`c_orig_balance_real = c_orig_balance × CPI(2025-01) / c_orig_cpi`，单位为 2025 年 1 月美元；CPIAUCNS 2025-01 = 317.671，即 [prepare](../prepare_turnover.py) 的 `CPI_BASE`，prepare 会校验 panel 中该月的 `c_orig_cpi` 与之相等。
`c_orig_cpi` 由 panel 按 origination month 查得，2025-10 未发布，按 9 月与 11 月插值。基准月只决定单位，不影响拟合。Nominal `c_orig_balance` 留在 frame 中作 diagnostics。
`c_factor = c_prev_balance / c_orig_balance`（t−1）替代 previous balance 作为 model input；previous balance 与 original balance 相关系数约 0.98，factor 与之约 0.12。
`c_sato = c_orig_rate − c_orig_pmms`，单位为百分点，基准为 origination month 的全国 PMMS，不含 LLPA；clip `[-1.5, 1.5]`，knots `-0.5, -0.25, 0, 0.25, 0.5, 1.0`。
Original LTV clip `[20, 97]`（>100 的 relief 类贷款截到 97），knots `50, 70, 80, 90, 95`。
注意 `updated LTV = original LTV × factor / HPI ratio` 恒成立（log 空间线性相关）。四者都进入模型时，FlatRamp 的分段线性仍可分辨它们，但单条曲线可能吸收其他三者的部分效应，解读时应同时看四条曲线。

Categorical blocks：purpose、occupancy、property type、first-time buyer、month、state，采用 OneHotEncoder(drop=None)。
Missing categories 为 MISSING；未知类别报错。Numeric missing rows 排除，不 impute。
Burnout、current balance/status 和 current macro 不作为 predictors；vintage 只作 diagnostics。Current balance 仍用于权重和 balance facets。

Polars 负责 filtering、target、categorical labels 和 artifacts。Clip 在 `to_model_data` 中完成，prepared frame 不保存 `_fit` 列。Model inputs 按列通过 NumPy 转入 pandas，不依赖 PyArrow。
每个 expanded block 先 cast float32 再拼接，避免 full-cohort one-hot matrix 被 numeric ramps 提升为 float64。

## 3. Report 与迭代

Fit 保存 model/toolkit、feature configuration、categories、cohort/fit 参数和 aligned predictions。
Frame / prediction hashes 在 report 时确认 artifacts 相互匹配；这是简单的输入一致性检查，不是 build cache。
Fit metadata 记录 fit 参数和 frame、prediction hashes，不记录个人绝对路径。Report 只替换 PDF，其他 artifacts 保持不变。

Report 展示收敛、components / implied actuals、actual vs predicted、reporting month 和 age×purpose。
Report 参考既有 turnover report，重建 saved toolkit 的 float32 design container，不重新 fit。
收敛、numeric/categorical implied actuals 直接调用 `LinearProductModelToolkit` 的标准绘图方法。
Actual-vs-predicted 直接使用 `MortgageDiagnostics` / `MortgageColnames`，包括 reporting month、incentive / age 及 purpose facets、LTV、factor 和 FICO。
HPI ratio、original LTV 没有 mortgage role，按源列名调用同一个 `MortgageDiagnostics.plot()`，不在 project 重写统计或绘图。
Implied actuals 与模型数值的 actual-vs-predicted 都使用 `*_fit`。Reporting month 和 current balance 没有 fit 列。
沿用 reference 的 CPR 显示：先聚合 bin-level SMM，再转换 `1-(1-SMM)^12`。Summary / context 仍保留月度 SMM。
两类 numeric/binned 图复用 `draw_binned_means()`，背景 Count bars 表示 loan-month 行数（不是余额或 unique loans），曲线/markers 使用左轴，不再以点大小表示 Count。
最低支持为 overall 500 rows、purpose facet 200 rows：低支持/空 bins 的曲线置空，Count bars 保留，曲线不跨空值连接。Purpose 与 balance facet panels 共用 Count 右轴和 CPR 主轴尺度，范围取自全量聚合，分页时各页一致；每个 panel 都显示两侧刻度数字，轴标题只在外侧显示一次。
Mortgage 图的 rounding 分箱沿用 `BinSpec.round`，聚合由 `summarize_binned_means` 以 Polars 表达式一次完成。Implied actuals 保留 toolkit 的 loss-specific 计算，小型 aggregated table 通过 `BinnedMeans.from_summary()` 传给 renderer。
CPR 转换在 bin-level SMM 聚合后执行，原数据及 predictions 不裁剪、不修改。日期使用真实日期位置，保留时间轴的月份间隔。

Poisson implied actual 使用现有 toolkit 的 convention：

```text
implied_actual = sum(weight × target / p) / sum(weight × m / p)
p = 当前 component；m = global scalar × 其他 components 的乘积
```

该 diagnostic 的除法由现有 toolkit 处理；raw predictions 不裁剪。
现有 solver 无 probability bounds，report 分别计数 nonfinite、negative、zero、above-one predictions。
若有负预测，严格 Poisson deviance 不定义，另列 solver 的 epsilon-floor deviance，并注明 fit 不是严格有效的 probability model。
所有 diagnostics 为 in-sample；下一阶段再做时间验证、interaction review 和 refinance component。

实现见 [prepare](../prepare_turnover.py)、[fit](../fit_turnover.py)、[report](../report_turnover.py)。
