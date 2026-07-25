# Run-Level Model Adapter Plan

## Goal

支持 independent example runs：不同 dataset、不同 coefficient set、不同 feature builder 可以各自成一个 runnable example，不需要改 `loan_simulation` core infra。

第一版目标是把现在的 GAM tie-out 往前推进一步：从 "直接喂 synthetic feature dict" 变成一个有清晰 layout 的 example run。

## Non-Goals

- 不把 dataset-specific feature mapping hardcode 进 `src/quantbullet/loan_simulation`。
- 不在第一版做 full cashflow simulation。
- 不在第一版处理 dials / overlays / production batch performance。
- 不把某一个 coefficient set 设计成 framework default；coefficient set 由 run config 指定。
- 不做 CSV / parquet dataset loader 抽象；有真实 dataset 需求时再加。

## Layering Rule

src = generic mechanics，examples = run-level semantics：

- Package layer 只放和任何 dataset 无关的机制：parse coefficients、callable 形状约定、assembly glue。
- Example layer 拥有所有 run-specific 语义：feature 映射、config、product assumptions、输出。
- 判定标准：任何具体 feature 名（如 `opti`、`v_ofico`、`month`）出现在 `src/` 里就是越界。

## Proposed Layout

Reusable adapter code 放在 package 里：

```text
src/quantbullet/loan_simulation/adapters/
  rollrate_gam.py           # TSV -> GAMReplayModel（已有）
  rollrate_bridge.py        # proposed: assembly glue，职责见下
```

`rollrate_bridge.py` 的职责边界（写死，不允许扩张）：

- feature builder 的形状约定：`Callable[[FeatureContext], dict[str, Any]]`。不做 formal protocol class，与现有 `EdgeSpec = float | Callable` 的风格保持一致。
- assembly glue：`edge_models + feature_builder + status_config -> SoftmaxTransitionModel`。即现在 `gam_tieout.py` 里 `build_quantbullet_model` / `_edge_logit_callable` 的通用化版本。
- 不放任何具体 feature 名、缺省 status universe 或 dataset 假设。

Specific runnable examples 放在 docs 下：

```text
docs/loan_simulation/examples/
  generic_v4_synthetic/
    README.md
    config.json
    feature_builder.py
    run.py
    .gitignore            # 忽略本 example 生成的 workbook
```

`examples/<example_name>/` 是 run-level ownership。每个 example 有自己的 dataset、feature mapping、coefficient set、product assumptions 和 output。换 dataset 或换 coefficient set 就新增 sibling folder，不改 core。

## Run Config Contents

`config.json` 除了 coefficient 位置，还要拥有 run-level 的产品假设（可以从 roll-rate config 初始化，但 example 自己保存一份）：

- coefficient set 与 root（如 `GENERIC_v4`）
- from statuses
- status universe / `status_to_roll`
- terminal statuses
- delinquency buckets
- seed、n_loans
- output workbook 路径（默认在 example folder 内）

这些目前 hardcode 在 `gam_tieout.py` 的脚本常量里（`TERMINAL_STATUSES`、`DELINQUENCY_BUCKETS`），对 diagnostic script 可接受，但 example run 必须由 config 拥有，否则第一个非 roll-rate 状态空间的 run 就要改 core。

## First Example: `generic_v4_synthetic`

- coefficient set: `GENERIC_v4`（真实 roll-rate coefficients）
- data: synthetic feature rows（第一版只做 synthetic；config 预留 `dataset` 字段）
- validation target: transition probabilities only

Expected output（写到 example folder 内，git ignored）：

- synthetic features
- raw logits by `from_status` / `to_status`
- softmax probabilities by transition row
- summary

Run from repo root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\generic_v4_synthetic\run.py --roll-rate-root C:\path\to\roll-rate-model
```

## Relationship With `gam_tieout.py`

- `gam_tieout.py` 保持 self-contained diagnostic：对照 roll-rate Python reference，自带一份小的 synthetic 生成器。
- tie-out script 不 import example，example 也不 import tie-out script；两处小重复可接受，docs scripts 之间不共享代码。
- 依赖方向永远是 docs -> src；不允许 src -> docs，也不允许 docs script -> example。

## Implementation Steps

1. 新增 `adapters/rollrate_bridge.py`：callable 约定 + `build_softmax_transition_model(...)` assembly glue，附 focused unit tests。
2. `gam_tieout.py` 改用 bridge 的 glue（行为不变，删除脚本内重复的 assembly 代码），但保留自己的 synthetic 生成器。
3. 创建 `docs/loan_simulation/examples/generic_v4_synthetic/`：`config.json`、`feature_builder.py`、`run.py`、`README.md`、`.gitignore`。
4. `run.py`：读 config -> parse coefficients -> bridge 组装 transition model -> 输出 transition probability workbook 到 example folder。
5. 跑通并核对输出后，更新 `docs/loan_simulation/README.md` 的 phase index（新增 example 条目）。

## Decisions (was Open Questions)

- Feature builder 用 simple callable convention，不做 protocol class。
- Example outputs 放各自 example folder 内，并被该 folder 的 `.gitignore` 忽略。
- 第一版只支持 synthetic data；`config.json` 预留 `dataset` 字段，不实现 loader。

## Deferred

- `required_features(replay_model) -> set[str]` 校验 helper：从 parsed term_data 推导每个 edge 需要的 feature 名，让 example 在 run 开始时 fail-fast。属于 generic mechanics（src），但不阻塞第一版。
- 真实 dataset 的 loader 与 mapping example。
- 接入 `LoanSimulator` 的 feature builder（`FeatureContext` 的 macro/path features 映射）。
