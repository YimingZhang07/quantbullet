# Design：LPM 的 form / solver 分层（identity BCD 与 log-link Newton 共存）

日期：2026-10-12 · 状态：设计草案，未实施 · 背景与实测见 [report](report.md)

## 目标与非目标

目标：

- 同一个模型类、同一套接口下，可选两种 block form：identity（现有 BCD）和 log（joint Newton，即 log-link Poisson GLM）。
- 每个变体的公式与限制集中在一个类中，fit、predict、toolkit 里不再按 loss 或 block 类型写 if/else 分支。
- 默认路径行为不变：数值结果不变，已保存的 pickle 可加载。
- 下游（toolkit、`MortgageDiagnostics`、`report_turnover`）不改或只改调用点。

非目标：

- 不支持 log + MSE（非凸）和 identity + Newton。
- 不改 `LinearProductClassifier*`；以后可作为 `IdentityBernoulli` form 接入（第 8 节）。
- 第一版不做 one-hot 结构化 Hessian 和 log form 的标准误。

## 1. 现状

模型为 `mu = global_scalar × prod_k f_k`，`f_k = X_k @ beta_k`，由 BCD 逐块更新，每块归一到均值 1，尺度并入 `global_scalar_`。
按 loss 和 block 类型的分支分散在多处（行号为 2026-10-12 的版本）：

| 分支 | 位置 |
|---|---|
| 按 loss | `regressor.py` 的 `loss_function`（21–30）、`_step_interaction_group`（207）、`_init_one_hot_codes`（252）、`_step_regular_group`（301）；`toolkit.py` 的 implied actual（356–361）、categorical panels（579–585）、收敛图标签（677） |
| 按 block 类型 | `interactions_` / `submodels_` 检查：`regressor.py` 23 处，`base.py` 5 处，`toolkit.py` 1 处 |
| 重复计算 block factor（`X @ coef`） | `base.py` 452、456、509；`regressor.py` 75、606、652；`toolkit.py` 296（toolkit 绕过了模型） |

外部使用者：`projects/freddie_prepayment/fit_turnover.py`、`tests/linear_product_model/` 下的 regressor、interaction、categorical_poisson 测试、
`tests/projects/test_freddie_turnover.py`，以及已保存的 `turnover_model.pkl`。

## 2. 三个变化轴

| 轴 | 变化内容 | 类 |
|---|---|---|
| **Form**（block link + loss） | 系数如何变成 factor、loss、归一化方式、implied actual 公式、标准误公式 | `IdentityMSE`、`IdentityPoisson`、`LogPoisson` |
| **Solver** | 迭代方式 | `BCDSolver`、`NewtonSolver` |
| **Block** | block 数据结构与求值 | `DenseBlock`、`OneHotBlock`、`InteractionBlock`、`FrozenBlock`（第二期） |

模型类只负责编排：解析 form、选 solver、保存结果，对外接口不变。

## 3. Form

```python
class ModelForm(ABC):
    """Link + loss: every formula that differs between variants, nothing about iteration."""
    name: str                    # "identity/poisson"; recorded in fit metadata and reports
    loss_label: str              # "Poisson deviance"
    default_solver: str          # "bcd" | "newton"

    def factor(self, X, coef): ...                        # block factor from its coefficients
    def rescale(self, coef, mean, constant_direction): ...  # coefficients whose factor has mean 1
    def loss(self, y, mu, w): ...                         # value tracked in loss_history_
    def implied_actual_terms(self, y, m, p, w): ...       # (numer, denom); implied = sum(numer) / sum(denom)


class BlockStepForm(Protocol):        # what BCDSolver needs
    def block_step(self, X, y, m, p, w, coef, codes=None): ...   # updated coefficients for one block


class NewtonForm(Protocol):           # what NewtonSolver needs
    def newton_terms(self, eta, y, w): ...   # (deviance, gradient weights, Hessian weights) per row
```

| | `IdentityMSE` | `IdentityPoisson` | `LogPoisson` |
|---|---|---|---|
| `factor` | `X @ coef` | `X @ coef`（loss 中 floor 到 1e-10） | `exp(X @ coef)` |
| `rescale` | `coef / mean` | `coef / mean` | `coef - log(mean) × c`，`c` 为 block 的常数方向 |
| `loss` | 加权 MSE | 加权 Poisson deviance | 加权 Poisson deviance |
| implied numer / denom | `w·m·y` / `w·m²` | `w·y/p` / `w·m/p` | `w·y` / `w·m` |
| solver 能力 | `block_step`：scaled OLS（含 QR/SVD 选项） | `block_step`：identity IRLS 一步，含 one-hot 分组求和 | `newton_terms` |
| 默认 solver | bcd | bcd | newton |

- **Implied actual 的统一原则**：三种 form 下都是 block 的 IRLS working response 在 bin 内按 working weights 的加权平均。
  working weights 与 response 只由 link 和 loss 决定，所以公式属于 form，toolkit 不再按 loss 分支。
- **常数方向 `c`**：满足 `X @ c = 1` 的系数方向。FlatRamp（`include_bias=True`）为 bias 列的单位向量，完整 one-hot 为全 1 向量。
  由 block 提供；没有常数方向的 block 在 log form 下拒绝（fit 开始时报错）。
- **Registry**：`resolve_form(block_link, loss)` 只登记上表三种组合，其他组合报错并列出支持的组合。

## 4. Solver

两个 solver 遵守同一个输出约定，fit 结束时必须设置：
`coef_`（ndarray 或 `InteractionCoef`）、`global_scalar_`、`block_means_`、`loss_history_`、`best_loss_`、`best_iteration_`、
`global_scalar_history_`、`global_scalar_step_history_`、`convergence_info_`、`fit_timing_`。模型保存 `form_` 和 `solver_name_`。

**`BCDSolver`**：从当前 `fit` 循环原样抽出，要求 form 实现 `block_step`。

- 初始化、每轮按 block 调用 `form.block_step`，用 `form.rescale` 归一化并把均值并入 `global_scalar_`。
- 保留 interaction 的 level 转移、ftol / early stopping、best iteration 恢复、`fit_timing_`。
- one-hot 快速路径由 form（`IdentityPoisson`）与 block 共同决定，solver 不判断 loss。

**`NewtonSolver`**：移植自 [compare_turnover.py](compare_turnover.py) 的 `LogLinkDesign` 与 `fit_poisson_newton`，要求 form 实现 `newton_terms`。

- 可识别的约简设计：数值 block 去掉 bias（全局一个 intercept），类别 block 去掉余额最大的 level，
  interaction 为 ramps × `by` 的 one-hot，所有列标准化；frozen submodel 进入 offset `log(prediction)`，要求预测为正。
- 分块计算，float32 design 缓存（turnover 约 10.6 GB），可选 stream；Hessian float32，gradient 与 deviance float64，带 step halving。
- 收敛后把约简系数还原到每个 block 的原始基（去标准化），用 `form.rescale` 归一到均值 1，
  interaction 在每个 category 内归一并把水平转入 `by` block，其余并入 `global_scalar_`。
  结果与 [fit_turnover_glm.py](fit_turnover_glm.py) 的转换相同，但系数是普通 ndarray，不需要 sklearn submodel。
- `loss_history_` 与 `global_scalar_history_` 按 Newton 步记录；`global_scalar_step_history_` 的 group 标为 `"newton"`。

## 5. Block（第二期）

| 类 | `factor(form, coef, X, cat)` | 常数方向 | 其他 |
|---|---|---|---|
| `DenseBlock` | `form.factor(X, coef)` | bias 列（若有） | |
| `OneHotBlock` | 同上，可用 codes 查表 | 全 1 | 提供 codes 给 BCD 快速路径 |
| `InteractionBlock` | 按 category 分别 `form.factor` | 每个 category 内的 bias | 持有 `by` 与 level 转移规则 |
| `FrozenBlock` | `submodel.predict(X)`，不经 form | — | BCD 跳过；Newton 作为 offset |

第二期之前，第一期先引入唯一入口 `model.block_factor(group, X_block, cat_series=None)`，第 1 节表中重复计算 factor 的位置全部改为调用它。
引入 Block 类后，`interactions_` / `submodels_` 的分支收进这些类，`block_factor` 只做转发。

## 6. 公共接口

```python
model = LinearProductRegressorBCD()
model.fit(container, feature_groups, interactions=..., loss="poisson", weights=w)                    # identity/poisson + bcd，同现在
model.fit(container, feature_groups, interactions=..., loss="poisson", weights=w, block_link="log")  # log/poisson + newton
model.fit(..., loss="mse", block_link="log")
# ValueError: unsupported form log/mse; supported: identity/mse, identity/poisson, log/poisson
```

- `fit` 新增 `block_link="identity"`；可选 `solver=None`（默认取 `form.default_solver`，显式指定时检查能力）。
  Newton 专用参数（`max_newton_iter`、`newton_tol`、`stream_design`、`chunk_rows`）通过 `solver_options` dict 传入，不扩展 `fit` 签名。
- BCD 专用参数（`n_iterations`、`early_stopping_rounds`、`force_rounds`、`ftol`、`cache_qr_decomp`、`use_svd`）保持原位置；
  在 newton 下传入非默认值时报错，避免静默忽略。
- `form_` 是 property，由 `loss_`（第 3 期起加上 `block_link_`）推出，不单独保存；新增 `solver_name_`；
  `fit_metadata_` 增加 `block_link` 与 `solver`。
- 兼容：form 不进 pickle，旧 pickle 无需迁移（缺 `block_link_` 时按 identity）；类名 `LinearProductRegressorBCD` 保留。

## 7. Caveat 的归属

| Caveat | 归属 |
|---|---|
| log + MSE 不支持 | `resolve_form`：无此组合，报错列出支持组合 |
| implied actual 公式 | `form.implied_actual_terms` |
| 归一化：除以均值 vs 平移常数方向 | `form.rescale` + block 的常数方向 |
| 负预测与 1e-10 floor | `IdentityPoisson.factor` / `loss` |
| one-hot 快速路径 | `OneHotBlock` codes + `IdentityPoisson.block_step` |
| frozen submodel 作为 offset，要求预测为正 | `NewtonSolver` + `FrozenBlock` |
| 无事件 level 的 MLE 发散 | `NewtonSolver` 的 `l2` 选项（默认 0）与警告 |
| design 缓存内存 | `NewtonSolver` 的 `stream_design` |
| 标准误 | form（公式）+ 持有 Hessian 的 solver；`IdentityPoisson` 现用 OLS 公式，标为已知问题 |
| 报告中标明模型 | `form_.name`、`solver_name_` 写入 metadata，`report_turnover` 的 Fit summary 显示 |

## 8. 下游改动

- **toolkit**：implied actual 与 categorical panels 改用 `model.form_.implied_actual_terms`；
  `compute_implied_actual_data` 改用 `model.block_factor`；收敛图 y 轴标签用 `form_.loss_label`；summary 显示 form 与 solver。
- **`MortgageDiagnostics`**：只读预测列，不改。
- **`fit_turnover`**（第五期）：config 增加 `block_link`，原样传给 `fit`；meta 记录 form 与 solver。
  `report_turnover` 的 Fit summary 显示 form 名，其余不改。
- **exploration**：实施后，[fit_turnover_glm.py](fit_turnover_glm.py) 的 sklearn submodel 适配不再需要。
- **classifier**（可选）：clipped cross-entropy 可成为 `IdentityBernoulli` form，`LinearProductClassifierBCD` 复用 `BCDSolver`。

## 9. 分期与验收

| 期 | 内容 | 验收 |
|---|---|---|
| 1 | `forms.py`（`IdentityMSE`、`IdentityPoisson`、registry）；`block_factor` 唯一入口；toolkit 改用 form | 行为不变：现有测试通过；固定种子的合成数据 fit 前后 `coef_`、预测、`loss_history_` 一致（≤ 1e-12）；turnover `--smoke-rows 100000` 的 metrics 一致；现有 `turnover_model.pkl` 可加载且预测一致 |
| 2 | 抽出 `BCDSolver` | 同第 1 期 |
| 3 | `LogPoisson` + `NewtonSolver` | 见下方测试；turnover 全量 `block_link="log"` 的 deviance 与 exploration 一致（0.04097777，±1e-9） |
| 4 | Block 类，收掉 `interactions_` / `submodels_` 分支 | 行为不变，同第 1 期 |
| 5 | turnover config 的 `block_link`；log form 标准误；one-hot 结构化 Hessian；可选 `IdentityBernoulli` | 各自测试 |

第 3 期测试：

- `LogPoisson` + Newton 与 sklearn `PoissonRegressor`（相同约简设计）结果一致。
- 只有类别 block 时两种 form 的模型空间相同：BCD（identity）与 Newton（log）的 deviance 一致。
- 收敛后，one-hot block 每个 level 的 implied actual 等于该 level 的 factor（score equations），对两种 form 分别成立。
- `rescale` 前后预测不变；pickle round trip；registry 拒绝不支持的组合；solver 能力检查。

## 10. 风险

- 第 1、2 期改动已测试的 BCD 数值代码：用逐期等价检查控制，每期单独提交、单独 review。
- pickle 兼容：`form_` 的 property 兜底，并用现有 `turnover_model.pkl` 做加载测试。
- Newton 内存：turnover 全量约多 10.6 GB；stream 模式更慢；结构化 Hessian 放在第 5 期。
- 两种 form 的曲线形状不同（LTV / factor / HPI 一组最多差 20%）：靠 metadata 与报告标注区分，避免混读。
- 范围膨胀：Block 类（第 4 期）与 classifier 可推迟，不影响第 1–3 期。

## 11. 决定与待决问题

已决定（2026-10-12）：

1. **参数名用 `block_link`**（`"identity"` / `"log"`）：它描述每个 block 的线性预测如何变成 factor；identity form 整体不是 GLM，
   所以不用模型级的 `link`。**不公开 `solver`**：每个 form 目前只有一个 solver，由 form 决定；
   solver 名记入 `solver_name_` 和 metadata，类属性 `bcd_solver` / `newton_solver` 可由子类替换。
   某个 form 有了第二个 solver 时再加 `solver=`。
2. **Newton 设置用 `solver_options` dict**，未知 key 报错并列出可用 key；BCD 参数保持原位以兼容。
   不复用 BCD 的参数名（`n_iterations` 数的是 sweep，`ftol` 比较 5 个 sweep 前的 loss，与 Newton 含义不同）。
   当前 solver 不使用的参数若取非默认值则报错，不静默忽略；turnover 在第 5 期按 form 传参（Newton 设置放在 `[fit.newton]`）。
3. **`design_cache="auto"` 为默认**，也可设 `True` / `False`。auto 在 float32 design（行 × 列 × 4 字节）不超过
   `max_cache_gb`（默认 16）时缓存，否则 stream。用固定上限而不是空闲内存，结果不随机器状态变化，也不新增依赖。
   所选模式记入 `fit_timing_` 与 metadata。缓存与 stream 只在 Hessian 精度上不同，gradient 都是 float64，收敛到同一最优点。

待决：

4. 第 4 期 Block 类是否做，或停在 `block_factor` 唯一入口。
5. Log form 的标准误范围：只给 ramp 与 level 系数，还是包括 bias。
6. Classifier 是否纳入 form 体系。
7. 实施完成后，exploration 下的 `compare_turnover.py`、`fit_turnover_glm.py` 保留还是删除。

## 12. 实施记录

**第 1 期**（branch `lpm-form-solver`，2026-10-12）：

- 新增 `forms.py`：`ModelForm`、`IdentityMSE`、`IdentityPoisson`、`resolve_form`、`form_of`；包内导出 `ModelForm`、`resolve_form`。
- `LinearProductRegressorBase`：`form_` property；`block_factor` 为唯一的 block 求值入口，`forward`、`predict`、
  `single_feature_group_predict`、`leave_out_feature_group_predict`、`_compute_block_means` 都经过它；`_predict_group` 保留为别名。
- `loss_function` 与 fit 的 loss 校验改用 form；不支持的 loss 报 registry 错误，模型保持原状。
- toolkit：implied actual、categorical panels、收敛图标签改用 `form_of(model)`；interaction 的 implied actual
  直接切分 block 预测，不再自行计算曲线；`_aggregate_implied_data` 的 `loss` 参数接受 form 或 loss 名。
- 新测试 `tests/linear_product_model/test_forms.py`（9 个）。

等价检查（turnover 10 万行样本；poisson / mse × 加权 / 不加权、frozen submodel、frozen interaction curve，以及已保存的
`turnover_model.pkl`）：325 个数组中，所有 fit 输出（系数、loss history、global scalar 轨迹、block means、implied actuals、
categorical aggregates）逐位相同；98 个 `predict` / `leave_out` 数组相差 ≤ 8.9e-14（相对数组尺度）。
原因是旧代码在 fit 中用 `np.dot`、在含 interaction 的 `predict` / `leave_out` 中用 `@`，二者在 column-major float32 block 上求和顺序不同；
现统一为 `np.dot`，fit 不受影响。已保存模型的预测与存档预测相差 1.5e-16。LPM 与 turnover 测试 97 通过；
2 个失败在改动前已存在（classifier BCD 测试传入 DataFrame 而 `memorize_fit_args` 读取 `X.response`；GPU 测试缺少 `cupy`）。

留给第 2 期：`regressor.py` 中 BCD block 更新内部的 3 处 loss 分支（interaction 更新、one-hot codes 初始化、regular 更新）
与 4 处直接的 identity 计算，随 `block_step` 移入 form、循环移入 `BCDSolver`。第 1 期已提交（`fce4a13`）。

**第 2 期**（同一 branch）：

- `forms.py`：identity forms 增加 `block_step`（MSE 为 scaled OLS，含 `qr` / `use_svd` 变体；Poisson 为 identity IRLS 一步，
  含 one-hot 分组求和）和能力声明 `grouped_one_hot`、`step_options`。代码逐行移入，计算不变。
- 新增 `solvers.py`：`BCDSolver` 负责初始化、block 更新、归一化、interaction 水平转移、记录、收敛判断与最优轮恢复，
  只接受带 `block_step` 的 identity form。BCD 只拟合 identity form，所以归一化（除以 block 均值）留在 solver；
  form 的 `rescale` 到第 3 期随 log form 引入。solver 是 `fit` 内的局部对象，不留在模型上。
- `LinearProductRegressorBCD.fit` 只做状态设置、输入校验和数据准备，迭代交给类属性 `bcd_solver`（子类可替换）；
  `regressor.py` 从 706 行降到 245 行，不再有按 loss 的分支。
- 加权的 QR / SVD 求解在迭代开始前报错（以前在第一个 block 更新时报，异常类型与信息相同）。
- 测试：调用私有方法的测试改为指向 `BCDSolver` 与 `form.block_step`；dense 参照模型改为替换 solver；
  新增 `test_solvers.py`（6 个）与 MSE `block_step` 测试。

等价检查：与第 1 期快照逐位相同（385 个数组；快照在第 1 期代码上补充了 MSE SVD、合成数据上的 QR、加权 QR / SVD 报错，
以及 turnover 数据上 QR 路径的既有错误）。LPM 与 turnover 测试 104 通过，2 个既有失败不变。

已知问题（既有，未改）：`cache_qr_decomp=True` 求解的是 `X @ coef ≈ y / scalar / fixed` 的未加权问题，
与 scaled OLS 不同；`fixed` 含 0 时 turnover 数据上 loss 变为 NaN，随后在收敛判断中报 `TypeError`。

第 2 期已提交（`eb2fbaf`）。

**第 3 期**（同一 branch，未提交）：

- `forms.py`：loss 改为逐行 `loss_terms` 加共用的加权平均（identity 结果逐位不变）；新增 `LogForm`（factor 为 `exp(X @ coef)`，
  `rescale` 沿常数方向平移）与 `LogPoisson`（`newton_terms`，implied actual 为 `sum(w·y) / sum(w·m)`）；registry 增加 `log/poisson`；
  每个 form 声明 `default_solver`。
- `solvers.py`：新增 `NewtonSolver` 与内部的 `_LogDesign`。design 由模型的 blocks 推出：完整 one-hot block 作类别（去掉余额最大的 level），
  其他 block 去掉全 1 列（若有）；interaction 要求 by block 是可训练的完整 one-hot block；frozen submodel 作为 offset `log(prediction)`，
  预测须为正。拟合后还原到各 block 的基并归一：有常数方向的 block 均值为 1，interaction 曲线在各 level 内均值为 1、水平移入 by block，
  没有常数方向的 block 保持原尺度（`block_means_` 记录其均值）。`loss_history_`、`global_scalar_history_`、`coef_history_` 按 Newton 步记录；
  `fit_timing_` 增加 `design_cache`、`design_columns`，`sweep_seconds` 为每次 pass 的时间。
- `fit`：新增 `block_link`、`solver_options`。按 form 的 `default_solver` 选 solver（类属性 `bcd_solver` / `newton_solver`）；
  每个 solver 用 `FIT_ARGUMENTS` 声明自己使用的 `fit` 参数，其他 solver 参数若取非默认值，在模型重置前报错（默认值从 `fit` 签名读取）。
  `fit_timing_` 改由 solver 建立。新属性 `block_link_`、`solver_name_`；`FitMetadata` 增加 `block_link`、`solver`。
- `calculate_feature_group_se` 对非 identity form 报 `NotImplementedError`。
- 暂不支持：log form 的 `interaction_submodels`（报错）、标准误、one-hot 结构化 Hessian。
- 测试：13 个新测试，包括与 sklearn 对照（含加权与 interaction）、frozen offset、全类别模型下两种 form 一致、optimum 处 implied actual
  等于 level factor、缓存与 stream 一致、参数检查、pickle、无事件 level 的警告。

验收：

- identity 路径与第 1 期快照逐位相同（385 个数组）；LPM 与 turnover 测试 117 通过，2 个既有失败不变。
- turnover 全量（17,042,421 行，156 列）`block_link="log"`：7 步收敛，余额加权 deviance 0.04097777197753082，与 exploration 相同；
  预测与 exploration GLM 预测的相对差 9.7e-15；所有 block 均值为 1；fit 262s（其中建 design 与缓存 51s），LPM 为 371–374s。
- `loss_history_` 沿用 identity Poisson 的 floor 公式，便于与 BCD 比较；对 y = 0 的行它与精确 deviance 相差约 8e-8（相对），
  精确值在 `convergence_info_['deviance']`。
