# 后续事项：LPM terms 与 turnover

日期：2026-10-22 · 状态：暂缓，以后处理 · 设计见 [design.md](design.md)，turnover 的 tent 对比见 [report.md](report.md)

tent basis 定下来时（turnover 改用 tent + log，不设 reference）留下、但决定暂不处理的事。设计上的待决问题仍以 design.md
第 12 节为准，这里只列出哪些还开着。某一项需要单独探索时（例如第 1 项），按 [explorations 约定](../README.md) 另开文件夹，
并在这里链接。

## 1. identity form（BCD）停在哪里取决于 basis

- **现象**：全量 turnover，identity/poisson，`turnover_identity.example.toml` 的设置（最多 60 个 sweep，ftol=1e-5）。
  FlatRamp 16 个 sweep 停下，deviance 0.0409796155；tent 24 个 sweep 停下，deviance 0.0409803138。两者的模型空间相同，
  但预测的最大相对差 80%，p99 13%，中位数 0.7%（report.md "六个版本"）。此前 FlatRamp 自己继续跑到 80 个 sweep，loss 只再降
  0.0009%，预测仍最多变 16.5%（[turnover.md](../../projects/freddie_prepayment/design/turnover.md) 第 2 节）。
- **含义**：identity 下 loss 已近平坦，曲线却没有定下来，相互依赖的 original LTV、factor、HPI ratio、updated LTV 尤其如此。
  所以 identity 的曲线不能拿来比较或下结论。log form 由 Newton 解到最优，不受影响。
- **可能的方向**：
  - BCD 的停止条件改看预测或系数的变化，不只看 loss；
  - 在 identity 下做联合 GN（design 6.4 节，P4；待决问题 6）；
  - 先确认这组曲线在 identity 下是否本来就弱可识别。
- **现状**：turnover 默认用 log，identity config 只当对照。`turnover_v1_identity` 的产物还是 FlatRamp 拟合的，也没有 `spec_`，
  要出报告得先重新 fit；处理本项时一起做。

## 2. turnover 的 reference

turnover 暂不设 reference，各 block 按 mean-1 归一化（待决问题 9）。以后要设 reference 时，需要决定：

- 每个特征各取一个常见值（report.md 的做法），还是取一组自洽的 reference state。updated LTV 恒等于
  original LTV × factor / HPI ratio，单独取值时四者对不上；
- 分类特征是否也设 reference；
- 报告怎样展示设了 reference 的 group（`block_means_` 不再是 1），implied actual 图上是否标出 reference 点（待决问题 2）。

## 3. 报告里的 knot 乘数表

tent 的系数就是各 knot 上的乘数，报告可以加一页 knot 乘数表（`model.term(name).table()`），比读曲线更直接。
这一项不影响拟合，可选。

## 4. 仍用旧 toolkit API 的 notebook

下面三个 notebook 还在用 `FeatureSpec` + `preprocess_config`，或者 `LinearProductModelToolkit(feature_config).fit(df)`。
P1(b) 之后 toolkit 只接收 `ModelSpec`，它们照原样跑不通。决定暂不修：

- [docs/linear_product_model/model_diagnostics.ipynb](../../docs/linear_product_model/model_diagnostics.ipynb)
- [docs/experiments/model-components-testing.ipynb](../../docs/experiments/model-components-testing.ipynb)
- [tests/linear_product_model/_test_interaction.ipynb](../../tests/linear_product_model/_test_interaction.ipynb)

## 5. 库里的小缺口

- `ColumnsTerm`（直接用已有列当 basis）列在 P1 范围里，没有实现，目前 `BasisTerm` 能覆盖它的用法。
- 常数方向仍由 Newton 按数据检测（`_constant_direction`），还没有改成"term 声明、按数据校验"（待决问题 8 的后半）。
- `name=` 只有 `TensorTerm` 有；`ByTerm` 的 `by` term 的名字必须等于它的列名。
- `ModelSpec` 的序列化以后再做（待决问题 18）。

## 6. 分期

- **P3 参数化 term**：暂缓（2026-10-22），等 turnover 出现具体需要联合拟合的参数形状再做（design 第 10 节）。
- **P4**：tent 的二阶差分惩罚与单调重参数化、identity 联合 GN、自由 knot，按需。

## 7. 设计上还开着的待决问题

design.md 第 12 节的 1（输出尺度）、2（第 2 项的报告展示）、4（`feature_groups=` 路径的去留）、5（`ByTerm` 的 theta 存储）、
6 和 7（属于 P3）、14（有意的约束）、15（不在 knot 上的多个 reference）、16（`ByTerm` 是否并入 `TensorTerm`）、
20（张量积的二维诊断）。
