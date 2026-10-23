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
| 2026-10-05 | balance × incentive 的二维修正面：offset 上的锚定双线性 D̂、C2 拉伸与 log-linear tilt 的对比、CrossTerm 工具计划 | 分析完成，2026-10-22 在当前产物上复现，未实施：[report.md](2026-10-05-balance-incentive-2d/report.md)。tilt 优于 C2，浅端有时代混淆。CrossTerm 计划见 [plan.md](2026-10-05-balance-incentive-2d/plan.md)，模型侧的二维 term 已由 lpm-terms 的 `TensorTerm` 实现 |
| 2026-10-12 | LPM BCD 与 log-link GLM：offset、速度、可替代性，form / solver 分层设计 | 已实施：`linear_product_model` 的 `forms` 与 `solvers`（`block_link="log"` 由 Newton 拟合），turnover 通过 `[fit] block_link` 选择。文件夹已删除，报告、设计与脚本见 commit `39365b0` 及之后的实施 commit |
| 2026-10-20 | LPM 的 Term 抽象：tent basis、reference 与 pivot、cross term（含张量积）、参数化 term、solver 的推广与分期 | 设计与部分实施：[design.md](2026-10-20-lpm-terms/design.md)，P0–P2 已实施，P3 起未实施。turnover 改用 tent 的对比见 [report.md](2026-10-20-lpm-terms/report.md)：拟合相同，是否采用待定 |
