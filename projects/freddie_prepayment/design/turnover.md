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
默认 60 sweeps、10-round early stopping、ftol=1e-8，不缓存 QR。

Prepared frame 保留 raw values。Fit 按 [fit](../fit_turnover.py) 的 `CLIP` 生成 `_fit` inputs：age、incentive、original FICO、updated LTV、original balance 和 HPI ratio。改 clip 或 knots 不重写 frame。
同一文件的 `KNOTS` 给出对应 FlatRamp knots。Implied-actual bin widths 在 [report](../report_turnover.py) 的 `IMPLIED_BIN_CONFIG`。
Updated LTV 是 first-lien estimate；`c_hpi_ratio = c_hpi_lag1 / c_orig_hpi`，表示 origination→lag1 的 ZHVI 倍数，不年化，也不是 trailing 24-month HPA。
`1.00` 表示持平，`1.10` 表示累计上涨 10%；fit clip 为 `[0.8, 2.5]`，knots 为 `0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0`，report bin width 为 `0.1`。
Original balance 为 nominal USD，本阶段不 inflation-adjust。

Categorical blocks：purpose、occupancy、property type、first-time buyer、month、state，采用 OneHotEncoder(drop=None)。
Missing categories 为 MISSING；未知类别报错。Numeric missing rows 排除，不 impute。
Burnout、current balance/status 和 current macro 不作为 predictors；factor、original LTV、vintage 只作 diagnostics。

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
Original balance、HPI ratio、original LTV 没有 mortgage role，按源列名调用同一个 `MortgageDiagnostics.plot()`，不在 project 重写统计或绘图。
Implied actuals 与模型数值的 actual-vs-predicted 都使用 `*_fit`。Reporting month、previous factor 和 original LTV 没有 fit 列。
沿用 reference 的 CPR 显示：先聚合 bin-level SMM，再转换 `1-(1-SMM)^12`。Summary / context 仍保留月度 SMM。
两类 numeric/binned 图复用 `draw_grouped_means()`，背景 Count bars 表示 loan-month 行数（不是余额或 unique loans），曲线/markers 使用左轴，不再以点大小表示 Count。
最低支持为 overall 500 rows、purpose facet 200 rows：低支持/空 bins 的曲线置空，Count bars 保留，曲线不跨空值连接。Purpose panels 共用 Count 右轴尺度。
Mortgage 图的 rounding 分箱沿用 `BinSpec.round`，聚合由 `summarize_grouped_means` 以 Polars 表达式一次完成。Implied actuals 保留 toolkit 的 loss-specific 计算，小型 aggregated table 通过 `GroupedMeansData.from_summary()` 传给 renderer。
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
