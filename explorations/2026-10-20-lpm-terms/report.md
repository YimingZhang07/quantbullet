# Report：turnover 改用 tent + reference，与 FlatRamp 的对比

日期：2026-10-21 · 状态：实验完成，是否采用待定 · 设计见 [design.md](design.md)（待决问题 9）

## 问题

turnover 的数值特征现在用 `BasisTerm(FlatRampTransformer(include_bias=True))`。改用 `TentTerm` 并设 reference 后：

1. 拟合是否不变；
2. 系数读起来是什么样子；
3. 耗时和内存是否变化。

## 结论摘要

- **是同一个模型。** 全量 17,042,421 行，log-link Poisson（Newton）。两者的 deviance 相对差约 3e-11，都在 7 步收敛，design 都是
  156 列。预测的最大相对差 4.4e-7，中位数 3.7e-8，这是 Newton 用 float32 Hessian 时的停止精度。knot 上的乘数两者一致，
  最大绝对差 5.5e-6。FlatRamp 的 deviance 和 best loss 与现有 `turnover_v1` 的保存值逐位相同。
- **耗时相同。** 准备数据 19 s 对 17 s，拟合 141 s 对 147 s。
- **差别只在读法。** tent 的每个系数就是曲线在该 knot 上的乘数，reference 处为 1（下文"乘数表"）。FlatRamp 的系数是各段斜率，
  要换算才能读。
- **`global_scalar_` 的含义变了。** FlatRamp 下它是各 block 均值为 1 时的水平（0.003876）。tent 下它是数值特征都取 reference、
  分类特征取平均水平时的月度 SMM（0.003595）。
- **建议采用**：改动只在 `fit_turnover.build_spec`，拟合不变。但 reference 怎么取最好先定下来，见"注意事项"第 1、2 条。

## 设置

- cohort 与现有 turnover 相同：`turnover.example.toml` 的 prepared frame，按余额加权，`block_link = "log"`，`[fit.newton]` 为
  `design_cache = "auto"`、`max_cache_gb = 16`，`l2 = 0`。
- FlatRamp：生产代码的 `build_spec()`，也就是 `CategoricalTerm` 加 `BasisTerm(FlatRamp)`，age 用按 purpose 分开的 `ByTerm`。
- tent：分类部分不变；每个数值特征换成 `TentTerm`，knots 为 `(clip 下界, *KNOTS, clip 上界)`。输入在 `to_model_data` 里已经截断，
  所以 tent 与 FlatRamp(+bias) 张成同一个曲线空间（design 5.1 节）。分类特征不设 reference，仍按 mean-1 归一化。
- reference 都取在 knot 上：age 24、incentive −1、SATO 0、FICO 740、original LTV 80、updated LTV 80、real original balance 300k、
  factor 0.9、HPI ratio 1.0。也就是一笔 24 个月、out of the money 1pp、SATO 0、FICO 740、LTV 80、30 万美元（2025 年币值）、
  余额为原始 90%、房价持平的贷款。

## 结果

| | FlatRamp | tent |
|---|---|---|
| Newton 步数 / 状态 | 7 / converged | 7 / converged |
| Poisson deviance（余额加权，同 turnover metadata） | 0.04097777197753 | 0.04097777197909 |
| design 列数 | 156 | 156 |
| 准备数据 / 拟合（秒） | 19 / 141 | 17 / 147 |
| 余额加权 A/E | 1.00000004 | 0.99999999 |
| 等权 A/E | 1.02819 | 1.02819 |
| 预测范围 | 0.000195–0.117 | 0.000195–0.117 |
| `global_scalar_` | 0.003876（mean-1） | 0.003595（reference） |

两者的预测相对差：最大 4.4e-7，p99 1.4e-7，中位数 3.7e-8。

tent 模型里，每个数值特征的 Newton pivot 都是它的 reference 列，例如 `c_age_fit_at_24`、`c_incentive_fit_at_-1`，
来源记为 `reference`。分类特征的 pivot 仍是权重最大的 level，来源记为 `auto`。

## 乘数表（tent，reference 处为 1）

`model.term(name).table()` 的输出，每格是该 knot 上的乘数。age 按 purpose 分开，每个 level 都在 24 个月处为 1，各 level 的整体水平
在 `f_purpose` 里。

**c_age_fit**

| level | 1 | 3 | 6 | 12 | 18 | 24 | 36 | 60 | 84 | 108 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 0.183 | 0.359 | 0.700 | 0.926 | 0.972 | 1.000 | 1.039 | 1.011 | 1.043 | 0.986 | 0.942 |
| N | 0.185 | 0.230 | 0.679 | 0.969 | 1.062 | 1.000 | 1.131 | 1.214 | 1.162 | 1.221 | 1.066 |
| P | 0.330 | 0.305 | 0.402 | 0.552 | 0.639 | 1.000 | 1.078 | 1.168 | 1.097 | 1.142 | 1.026 |

**c_incentive_fit**

| -5 | -4 | -3 | -2 | -1.5 | -1 | -0.75 | -0.5 |
|---|---|---|---|---|---|---|---|
| 0.337 | 0.498 | 0.609 | 0.785 | 0.955 | 1.000 | 1.051 | 1.038 |

**c_sato_fit**

| -1.5 | -0.5 | -0.25 | 0 | 0.25 | 0.5 | 1 | 1.5 |
|---|---|---|---|---|---|---|---|
| 0.895 | 0.937 | 0.987 | 1.000 | 1.028 | 1.082 | 0.980 | 1.014 |

**c_orig_fico_fit**

| 620 | 660 | 700 | 740 | 780 | 800 |
|---|---|---|---|---|---|
| 1.075 | 1.025 | 1.015 | 1.000 | 0.925 | 0.865 |

**c_orig_ltv_fit**

| 20 | 50 | 70 | 80 | 90 | 95 | 97 |
|---|---|---|---|---|---|---|
| 0.820 | 0.907 | 0.961 | 1.000 | 1.149 | 1.243 | 1.184 |

**c_updated_ltv_fit**

| 5 | 20 | 40 | 60 | 80 | 95 | 100 |
|---|---|---|---|---|---|---|
| 1.979 | 1.357 | 1.147 | 1.216 | 1.000 | 0.604 | 0.439 |

**c_orig_balance_real_fit**

| 30k | 125k | 200k | 300k | 425k | 600k | 850k | 1000k |
|---|---|---|---|---|---|---|---|
| 1.282 | 1.168 | 1.139 | 1.000 | 0.960 | 0.931 | 0.866 | 0.774 |

**c_factor_fit**

| 0.2 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | 1 |
|---|---|---|---|---|---|---|
| 3.735 | 1.059 | 0.966 | 0.943 | 1.000 | 1.205 | 1.585 |

**c_hpi_ratio_fit**

| 0.8 | 0.95 | 1 | 1.1 | 1.25 | 1.5 | 1.75 | 2 |
|---|---|---|---|---|---|---|---|
| 4.539 | 0.923 | 1.000 | 1.123 | 1.256 | 1.317 | 1.310 | 1.149 |

这些表让"相邻 knot 之间变化多少"一眼可见。例如 factor 从 0.5 到 0.2、HPI ratio 从 0.95 到 0.8 两段，乘数分别变了约 3.5 倍和
4.9 倍；之前量过，几何插值和线性插值的差别也集中在这两段。这两段都在数据稀疏的边缘，乘数主要由少量样本决定。

## 六个版本：log / identity × FlatRamp / tent（mean-1）/ tent（reference）

每个版本都跑了正常的 turnover fit 和 report，输出在各自的文件夹 `${FREDDIE_DATA_ROOT}/models/turnover_variants/{版本}`。
cohort 相同，用同一个 prepared frame；identity 版本用 `turnover_identity.example.toml` 的 BCD 设置（最多 60 个 sweep，
early stopping 10，ftol 1e-5）。

| 版本 | form | 迭代 | 拟合（秒） | deviance（余额加权） | 余额加权 A/E | 等权 A/E | `global_scalar_` |
|---|---|---|---|---|---|---|---|
| tent_reference_log | log/poisson | 8 | 187 | 0.0409777720 | 1.000000 | 1.028193 | 0.003595 |
| tent_mean1_log | log/poisson | 8 | 151 | 0.0409777720 | 1.000000 | 1.028193 | 0.003876 |
| flat_ramp_log | log/poisson | 8 | 134 | 0.0409777720 | 1.000000 | 1.028193 | 0.003876 |
| tent_reference_identity | identity/poisson | 24 | 489 | 0.0409803138 | 1.000000 | 1.029156 | 0.003367 |
| tent_mean1_identity | identity/poisson | 24 | 567 | 0.0409803138 | 1.000000 | 1.029156 | 0.004061 |
| flat_ramp_identity | identity/poisson | 16 | 384 | 0.0409796155 | 1.000000 | 1.030358 | 0.003842 |

同一种 link 下，逐行比较预测（tent 对 FlatRamp）：

| link | 比较 | 最大相对差 | p99 | 中位数 |
|---|---|---|---|---|
| log | tent_reference vs flat_ramp | 4.41e-07 | 1.45e-07 | 3.74e-08 |
| log | tent_mean1 vs flat_ramp | 4.40e-07 | 1.44e-07 | 3.61e-08 |
| identity | tent_reference vs flat_ramp | 8.04e-01 | 1.33e-01 | 7.33e-03 |
| identity | tent_mean1 vs flat_ramp | 8.04e-01 | 1.33e-01 | 7.33e-03 |

- **log 下三个版本是同一个模型。** 不设 reference 时，tent 的 `global_scalar_` 和曲线都和 FlatRamp 一样（0.003876）；设了
  reference，只是每条曲线整体乘一个常数，`global_scalar_` 反向吸收（0.003595）。三者的 Newton 步数相同；拟合时间不同，这次没有
  重复测量，不能据此比较速度。
- **identity 下，有无 reference 的 tent 是同一个拟合**（deviance 和预测都相同），reference 同样只是换口径。
- **identity 下 tent 和 FlatRamp 不一样**，但这不是 tent 造成的。两者的模型空间相同，区别在 BCD：它从各自 basis 的初值
  （`init_betas_by_response_mean`）出发，又没有收敛就停了（turnover.md 记录过：ftol 停止时相关曲线仍在移动），所以停在不同的点。
  这次 FlatRamp 16 个 sweep 停下，deviance 更低（0.0409796 对 0.0409803）；tent 24 个 sweep 才停下。两者预测的中位数差 0.7%，
  最大差 80%。log link 由 Newton 解到最优，没有这个问题。

## 注意事项

1. **reference state 不自洽。** `updated LTV = original LTV × factor / HPI ratio` 恒成立；按上面的 reference，updated LTV 应为
   80 × 0.9 / 1.0 = 72，而它的 reference 取了 80。每个乘数仍然是"相对自己的 reference"，读法不受影响；但 `global_scalar_`
   对应的是一笔不存在的贷款。如果要把 `global_scalar_` 读成某类贷款的速度，四者的 reference 需要一起取，且最好都落在 knot 上。
   reference 只挪动整体水平，换 reference 不改变拟合：本次对 FlatRamp 也是拟合后按 reference 换算得到乘数表。
2. **分类特征没设 reference。** `global_scalar_` 里混着分类特征的平均水平。要得到干净的 reference state，可以给 purpose、
   occupancy、property type、first-time buyer、month、state 也设 reference。
3. **`l2 > 0` 时两种 basis 不等价。** Newton 的 `l2` 惩罚的是标准化后的系数，和 basis 有关。在 10 万行样本上（有些 level 没有
   事件，需要 `l2 = 1e-8`），两者预测的 p99 相对差达到 0.96，差别集中在没有事件的 level。全量拟合用的是 `l2 = 0`，不受影响。
4. **端外行为不同。** tent 在 clip 边界外是平的，FlatRamp 是线性外推。现在输入先截断，所以两者没有区别；如果以后不再截断，
   两个模型在边界外会不同。
5. **现有 artifacts 都要重新 fit。** P1(b) 之后，model 要带 `spec_` 才能出报告；不论是否改用 tent 都需要这一步。

## 复现

从仓库根目录运行。结果写到 `--out`，必须在仓库之外：

```bash
.venv/Scripts/python.exe explorations/2026-10-20-lpm-terms/compare_tent_turnover.py --config projects/freddie_prepayment/turnover.example.toml --out "${FREDDIE_DATA_ROOT}/models/turnover_tent_compare/full"
```

六个版本：每个版本跑一次 `report_turnover_variant.py`（`--basis tent|flat_ramp`，tent 可加 `--mean-one`；identity 版本用
`turnover_identity.example.toml`），再汇总：

```bash
.venv/Scripts/python.exe explorations/2026-10-20-lpm-terms/report_turnover_variant.py --config projects/freddie_prepayment/turnover.example.toml --out "${FREDDIE_DATA_ROOT}/models/turnover_variants/tent_reference_log" --basis tent
```

```bash
.venv/Scripts/python.exe explorations/2026-10-20-lpm-terms/summarize_turnover_variants.py "${FREDDIE_DATA_ROOT}/models/turnover_variants"
```

对比脚本输出 `summary.json`（各项指标和 pivot）与 `knot_tables.csv`（两种 basis 在每个 knot 上的乘数）。小样本用
`--smoke-rows 100000 --l2 1e-8`，但按注意事项第 3 条，小样本只能检查流程，不能检查两者是否等价。

## 待决问题

1. 是否把 turnover 的数值特征改成 tent。改动只在 `build_spec`，拟合不变。
2. reference 怎么取：每个特征各自取一个常见值（本次的做法），还是取一组自洽的 reference state；分类特征是否也设。
