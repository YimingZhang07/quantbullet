# Freddie prepayment modeling

从 Freddie SFLLD 和 housing macro 数据刷新本项目的 prepared panel，再构建 turnover cohort，fit multiplicative model，并独立生成 diagnostics PDF。
`refresh` 是本项目的协调入口：按顺序触发通用的 [Freddie](../../procs/freddie_sflld/README.md) 和 [housing macro](../../procs/housing_macro/README.md) procs，以及本项目的 turnover 阶段。

## Setup

在 repository root 使用项目 `.venv`。在仓库根 `.env` 设置仓库外的数据目录（见 `.env.example`）；本项目入口会自动读取 `.env`，shell 中显式设置的变量优先。

```
FREDDIE_DATA_ROOT=<external-data-root>/freddie_sflld
MACRO_DATA_ROOT=<external-data-root>/housing_macro
```

Freddie 原始 ZIP 需手动放到 `$FREDDIE_DATA_ROOT/raw/full_set_standard_historical_data.zip`。

| Config | 内容 |
| --- | --- |
| [sample_panel.example.toml](sample_panel.example.toml) | 抽样规格：2010Q1–2026Q1 的 30 年 FRM，800k loans，输出 `samples/prepayment_800k` |
| [prepare_panel.example.toml](prepare_panel.example.toml) | sampled panel、macro 和 prepared panel 目录 |
| [turnover.example.toml](turnover.example.toml) | 输入 panel、turnover 输出目录、cohort 与 fit 参数（`block_link = "identity"`，BCD） |
| [turnover_log.example.toml](turnover_log.example.toml) | 同上，`block_link = "log"`：log-link Poisson GLM（Newton），设置在 `[fit.newton]`，输出到独立目录 |

三份 config 必须首尾相接：panel 的 `sample_root` 等于 sample 的 `output_root`，turnover 的 `panel_path` 是 panel `output_root` 下的 `panel.parquet`。`refresh` 运行前检查，不一致时不执行任何 stage。

## 执行顺序

| Stage | 运行 | 产物 |
| --- | --- | --- |
| `macro` | `housing_macro.download --refresh` → `housing_macro.build_parquet` | `$MACRO_DATA_ROOT/parquet/` 下的 `hpi`、`cpi`、`pmms` |
| `convert` | `freddie_sflld.build_parquet`，vintage 范围取自 sample config | `$FREDDIE_DATA_ROOT/parquet/`、`manifests/conversion.json` |
| `sample` | `freddie_sflld.build_sample_panel` | sample config 的 `output_root` |
| `panel` | `freddie_sflld.prepare_panel` | panel config 的 `output_root/panel.parquet` |
| `frame` | `prepare_turnover` | `turnover_frame.parquet` |
| `fit` | `fit_turnover` | `turnover_model.pkl`、`turnover_predictions.parquet` |
| `report` | `report_turnover` | `turnover_report.pdf` |

```powershell
.\.venv\Scripts\python.exe -m projects.freddie_prepayment.refresh --from panel
```

`--from` 必填，`--to` 默认 `report`。每个 stage 在独立进程中运行；任一步失败即停止，后续 stage 不运行。`--dry-run` 只打印命令。
使用其它 config 时传 `--sample-config`、`--panel-config`、`--turnover-config`。各 stage 覆盖固定输出，同一组目录一次只运行一个 refresh。

## 迭代

| 改动 | 运行 |
| --- | --- |
| report 的 plots、bins 或 layout | `--from report` |
| clip、knots 或其它 fit 参数 | `--from fit` |
| 另一种模型形式（`block_link`） | 用对应的 turnover config；它的输出目录第一次运行 `--from frame`，之后 `--from fit` |
| cohort、target、labels 或进入 frame 的 fields | `--from frame` |
| 上游 feature 定义或列名 | `--from panel`；旧 panel、frame 和 saved model 不自动转换 |
| macro 数据更新 | `--from macro --to macro`，再 `--from panel` |
| 抽样规格 | `--from sample` |
| 新的 Freddie release | 替换 ZIP、按需调整 sample config 的 `end_vintage`，再 `--from convert`；未变的 vintage 自动复用 |

- Report 不调用 prepare / fit；缺少或不匹配的 artifacts 会报错。
- 单个 stage 也可直接运行模块。`fit_turnover --smoke-rows 100000` 显式进行诊断抽样，需配合使用独立 output directory 的 turnover config；默认 fit 全量，不抽样。

方法、字段与边界见 [design/turnover.md](design/turnover.md)；panel feature 见 [panel preparation](../../procs/freddie_sflld/design/panel_preparation.md)。
