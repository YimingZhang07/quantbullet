# Explorations

探索性分析、方案评估和 further improvements 的记录。这里的结论尚未实施，或用于支撑实施前的决策；
正式的设计和规则仍放在对应 project / procs 的文档中。

## 约定

- 每个主题一个文件夹 `YYYY-MM-DD-<topic>/`，日期为开始日期。
- 文件夹内 `report.md` 为主报告：问题、结论、证据、待决问题和复现命令。
- 脚本与报告放在同一文件夹，从 repo root 用 `.venv` 运行。只用合成数据，或通过环境变量读取本地数据；
  不提交数据、生成文件或个人路径。
- 脚本不是 tests：不放进 `tests/`，pytest 也不会收集（`testpaths = ["tests"]`）。
- 结论被实施或否决后，更新下方状态，并链接到对应 commit 或设计文档。

## 索引

| 日期 | 主题 | 状态 |
|---|---|---|
| 2026-10-12 | [LPM BCD 与 GLM：offset、速度与可替代性](2026-10-12-lpm-vs-glm/report.md)；[form / solver 分层设计](2026-10-12-lpm-vs-glm/design.md) | 待评估；turnover in-sample 对比已完成，时间外待跑；设计草案待决 |
