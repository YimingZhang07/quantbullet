# Freddie prepayment modeling

从 prepared panel 构建 turnover cohort，fit multiplicative model，再独立生成 diagnostics PDF。
在 repository root 使用项目 `.venv`。在仓库根 `.env` 设置 `FREDDIE_DATA_ROOT` 为仓库外的 Freddie 数据目录；命令本身不用再导出变量。
配置见 [turnover.example.toml](turnover.example.toml)。

## 执行顺序

| 阶段 | Module | 产物 |
| --- | --- | --- |
| 1. Prepare | `prepare_turnover` | `turnover_frame.parquet` |
| 2. Fit | `fit_turnover` | `turnover_model.pkl`、`turnover_predictions.parquet` |
| 3. Report | `report_turnover` | `turnover_report.pdf` |

```powershell
.\.venv\Scripts\python.exe -m projects.freddie_prepayment.prepare_turnover --config projects/freddie_prepayment/turnover.example.toml
.\.venv\Scripts\python.exe -m projects.freddie_prepayment.fit_turnover --config projects/freddie_prepayment/turnover.example.toml
.\.venv\Scripts\python.exe -m projects.freddie_prepayment.report_turnover --config projects/freddie_prepayment/turnover.example.toml
```

所有产物保存到 config 的外部 `output_root`，使用固定文件名。

## 迭代

- 修改 report 的 plots、bins 或 layout：只运行第 3 步。
- 修改 clip、knots 或其它 fit 参数：从已有 frame 运行第 2、3 步。
- 修改 cohort、target、labels 或进入 frame 的 fields：运行全部三步。
- 修改上游 feature 定义或列名：先运行 [Freddie `prepare_panel`](../../procs/freddie_sflld/README.md)，再运行全部三步；旧 panel、frame 和 saved model 不自动转换。
- Report 不调用 prepare / fit；缺少或不匹配的 artifacts 会报错。
- `fit_turnover --smoke-rows 100000` 显式进行诊断抽样；使用独立 output directory。默认 fit 全量，不抽样。

方法、字段与边界见 [design/turnover.md](design/turnover.md)。
