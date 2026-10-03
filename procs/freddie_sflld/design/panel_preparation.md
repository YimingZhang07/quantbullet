# Prepayment data preparation 与 feature construction

这份文档解释从 Freddie 原始数据到 enriched loan-month panel 的过程，以及各个派生字段的构造逻辑。
运行命令和配置见[流程 manual](../README.md)。这里记录当前实现，方便对照代码理解和维护。

## 1. 数据流程

```text
Full Standard Dataset ZIP
  → 按 vintage 转换为独立 orig / perf Parquet
  → product filtering + vintage stratified loan sampling
  → sampled static / dynamic join
  → 日期和编码清洗、previous fields、macro matching、derived features / states
  → 单个 enriched panel.parquet + preparation_summary.json
```

### 上游 conversion 与 sampling

Conversion 保留 source 字段的 String 类型和官方编码，blank 转为 null；orig 和 perf 分开按发放季度保存。
输入文件通过 `manifests/conversion.json` 定位，避免读到更新过程中可能共存的旧文件。
Manifest 记录 source/output hashes；hash 未变化的季度可以跳过转换。原始 ZIP 保持不变，每次只解压和处理一个季度。
原始字段定义见 [Freddie Release 47 guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf)。

Sampling 的单位是 **unique loan**。先对 orig 应用可选的 `amortization_type` 和 `original_loan_term` filters，
再按各 vintage 的 eligible loan 数比例分配 quota。两个 filters 同时设置时用 AND；term 按月数做 numeric comparison。
Quota 使用 largest remainder：先取整，再按余数补齐，余数相同按 vintage 升序分配。
每季度先排序 loan IDs，再用 `seed:vintage` 做无放回抽样；同一 Python environment、输入贷款集合和 config 下可复现。
例子 config 选择 2015Q1–2026Q1 的 500k、360-month FRM loans；实际范围与样本数由 TOML 决定。

抽样不根据是否 prepay、default、存续多久或是否有 perf 决定是否入选。每笔样本保留完整已报告 perf history，
再按 `loan_identifier` many-to-one join static。没有 perf 的贷款仍在 `sampled_loans.parquet`，但不补造 loan-month rows：

```text
sampled loans = panel unique loans + sampled loans without perf
```

采样产物有 65 个合并 source 字段，再增加 `vintage` (String) 和 `month` (Date)。
Source `period` 和 missing codes 都保留。Summary 区分 filter 前的 `raw_population_loans`、filter 后的 `population_loans`，
并记录 quota、sample fractions 和有/无 perf 的贷款数。只有完成整个采样流程才写出成功 summary。
具体算法见 [sampling.py](../../../src/quantbullet/data/freddie_sflld/sampling.py)。

### Preparation 的 inputs 与处理顺序

| Input | 用途 |
| --- | --- |
| `sample_root/sampled_loans.parquet` | 每笔样本的 static attributes |
| `sample_root/panel/vintage=.../panel.parquet` | 已 join static 的原始 loan-month records |
| `sample_root/sampling_summary.json` | 确定 vintages，核对贷款数和行数 |
| `macro_root/parquet/hpi.parquet` | Zillow state / national ZHVI |
| `macro_root/parquet/pmms.parquet` | 全国月度 30-year PMMS |
| `macro_root/parquet/cpi.parquet` | 全国月度 CPIAUCNS |

当前 [prepare_panel.py](../prepare_panel.py) 的执行顺序是：

1. `prepare_macro_tables()` 选择所需 series 和 HPI geography，检查 macro keys 与数值。
2. `derive_loan_features()` 对 sampled static 清洗并计算 loan-level fields（含原合同参考月供），关联 origination PMMS 和 HPI candidates。
   这一步只做一次；小型 macro tables 和贷款级 features 可以放在内存。
3. 逐 vintage 读取 panel，检查 sample membership、行数和 unique loan-month keys。
4. `prepare_loan_months()` 按 `loan_identifier, month` 排序，计算连续上月字段、还款构成、累计 modification、status 和 quality flags，
   再关联 monthly macro、选择 HPI geography 并计算 derived features。
5. 每季度通过 LazyFrame / `sink_parquet()` 写临时结果，按 `d_reporting_month` 排序后 streaming merge 成一个 Parquet。
   最终只发布 `panel.parquet` 和 `preparation_summary.json`，临时季度文件清理掉。

Loan history 的 window calculation 在各 vintage 内完成，因此不需要把全量 panel 载入内存。
最终文件按 reporting month 排序，row groups 集中相近月份，方便 downstream 按月份读取。
全部原始列和行保留，内部 `_...` helper columns 不输出。实现见
[features.py](../../../src/quantbullet/data/freddie_sflld/features.py)。

Missing/duplicate keys、非法日期、不可解析或非有限数值会使构建失败；正常 null 允许保留。
Macro joins 使用 many-to-one validation，输出行数必须与输入一致。
整次构建先写 staging directory，成功后替换专用 output；转换或校验失败时保留上一版成功产物。
Summary 记录行数、贷款数、feature null fractions、status/exit counts、HPI fallback、quality counts 和 `ever_modified_rows`。

## 2. 日期与 naming

| Prefix | 类型与用途 |
| --- | --- |
| `d_` | Date，统一为月初 |
| `c_` | Numeric；`c_age` 为 Int32，其余为 Float64 |
| `f_` | Categorical String，包括 units / borrowers 等计数类别 |
| `is_` | Boolean，描述 continuity、observed modification 或 data quality |

`orig`、`prev`、`current`、`updated` 放在变量主体前；`lag1` 放在末尾。
新增列不覆盖原始字段，例如 `classic_fico` 保持 String，另存清洗后的 `c_orig_fico`。

| 字段 | Source / calculation | Missing rule |
| --- | --- | --- |
| `d_first_payment_month` | 解析 `first_payment_date` 的 YYYYMM | Blank/null → null |
| `d_origination_month` | `d_first_payment_month - 1 month` | First payment month 缺失 → null |
| `d_maturity_month` | 解析披露的 `maturity_date` | Blank/null → null |
| `d_reporting_month` | 原始 panel 的 `month` | 必须非空且为月初 |
| `d_exit_month` | 解析 `zero_balance_effective_date` | Blank/null → null，不根据 status 补日期 |
| `c_age` | `12 × (reporting year - origination year) + reporting month - origination month` | Origination month 缺失 → null |

`d_origination_month` 是推定月份，不是披露的 closing date。`c_age` 以这个起点计算，modification 后不重置；
原始 `loan_age` 同时保留。Age 为 0、负数或缺失的记录也保留，由 downstream 判断如何使用。
非法的非空 YYYYMM 会报错，不会静默变成 null。

### Previous observation 与 macro lag1

对 reporting month **t**，贷款 previous fields 只有在同一贷款确实存在 **t−1 日历月**记录时才有值。
第一条记录或月份断档时为 null；不拿 original balance 填补，也不把上一条更早记录当成 t−1。

Macro `lag1` 直接 join **t−1 的 macro month**，不依赖贷款是否有 t−1 记录。
因此月份断档时，`c_prev_balance` 可能为空，但 `c_pmms_lag1` 仍有值。
月份偏移只用于内部 lookup，不输出 `d_feature_month`。

## 3. Feature construction

所有清洗后的 text 先 trim whitespace，blank 和已有 null 保留为 null；另按字段处理下面列出的 sentinel codes。
Numeric cast 为 Float64，非空非法数值报错。Source columns 不改；derived categories 保留原始类别编码，不做 one-hot encoding。

### Original numeric attributes

以下 attributes 来自 orig record，时点为 origination，不随 reporting month 更新。

| Feature | Source | 单位 | 额外 missing code |
| --- | --- | --- | --- |
| `c_orig_fico` | `classic_fico` | FICO score | `9999` |
| `c_orig_balance` | `original_upb` | USD | 无 |
| `c_orig_term` | `original_loan_term` | Months | 无 |
| `c_orig_rate` | `original_interest_rate` | Percent | 无 |
| `c_orig_ltv` | `original_ltv` | Percent | `999` |
| `c_orig_cltv` | `original_cltv` | Percent | `999` |
| `c_orig_dti` | `original_dti` | Percent | `999` |

### Original categorical attributes

| Feature | Source | 额外 missing code |
| --- | --- | --- |
| `f_purpose` | `loan_purpose` | `9` |
| `f_occupancy` | `occupancy_status` | `9` |
| `f_property_type` | `property_type` | `99` |
| `f_state` | `property_state` | 无 |
| `f_channel` | `channel` | `9` |
| `f_first_time_buyer` | `first_time_homebuyer_indicator` | `9` |
| `f_units` | `number_of_units` | `99` |
| `f_borrowers` | `number_of_borrowers` | `99` |
| `f_vintage` | `vintage` | 无 |

字段定义与 missing codes 对应 `ORIG_NUMBERS` / `ORIG_FACTORS`。
其中每项的结构是 `raw_column: (derived_column, missing_codes)`；空 tuple 表示没有额外 sentinel code。

### Previous 与 reporting-month attributes

| Feature | Source / calculation | 时点、类型 / 单位与 missing rule |
| --- | --- | --- |
| `c_balance` | `current_actual_upb` 清洗并 cast 后的 alias | t，Float64 / USD；blank/null → null，原始 String 列保留 |
| `c_prev_balance` | 上月 `current_actual_upb` | t−1，USD；无连续上月记录或 source 缺失 → null |
| `c_prev_rate` | 上月 `current_interest_rate` | t−1，Percent；同上 |
| `f_prev_modified` | 上月 `modification_flag` 的归一值 | t−1，String；`Y`/`P` → `Y`，blank/null → `N`，其他 → `UNKNOWN`；无连续上月记录 → null |
| `f_pre_status` | 上月 `f_status` | t−1，String；无连续上月记录 → null，mapping 见第 4 节 |
| `f_month` | `d_reporting_month` 的月份 | t，String `01`–`12` |

Previous balance / rate 反映 modification 后的实际字段。这里不要求 previous balance 必须为正，也不因为 previous status 未知而删行。

### Reference payment 与还款构成

当前 sample 为 fully amortizing FRM。根据原合同参数计算固定 monthly P&I reference payment：

```text
P = c_orig_balance
n = c_orig_term
r = c_orig_rate / 1200

c_monthly_payment = P × r / (1 − (1 + r)^(-n))
r = 0 时：c_monthly_payment = P / n
```

Rates 使用 percent units，所以 `4.0 / 1200` 才是 monthly decimal rate。
P 或 n 缺失或非正、original rate 缺失或为负时，payment 为 null。月供只含 principal / interest，不含 escrow、税费和保险。
该值在 loan-level 计算一次，modification 后仍保持原合同参考值。

| Feature | Formula | 时点、单位与 missing rule |
| --- | --- | --- |
| `c_monthly_payment` | 上述 original-contract annuity formula | Original reference，Float64 / USD；无效参数 → null |
| `c_interest` | `c_prev_balance × c_prev_rate / 1200` | t 的 interest estimate，Float64 / USD；previous inputs 缺失 → null |
| `c_scheduled_principal` | `c_monthly_payment − c_interest` | t 的 scheduled principal estimate，Float64 / USD；任一输入缺失 → null |
| `c_scheduled_balance` | `c_prev_balance − c_scheduled_principal` | t 的 expected ending balance，Float64 / USD；任一输入缺失 → null |

Scheduled balance 是从 **上月 actual balance** 出发，只支付 reference payment 后的一步预测，不是从 origination 开始的原始摊还曲线。
首行或月份断档时，三个 monthly split fields 都为 null，reference payment 仍保留。不使用 current balance / rate 代替 previous inputs。
金额不取整、不截断正负值；退出后等记录也保留公式结果，解释与筛选由 downstream 决定。本阶段不计算 curtailment 差额。

这些是 **estimates**，不是披露的 payment / scheduled UPB：Freddie original UPB 取整到最近 $1,000；
未修改贷款的 source `loan_age <= 6` 且 actual UPB > $500 时，actual UPB 也按 $1,000 取整（见第 1 节官方 guide）。
Modification 后参考月供可能不再对应实际合同；interest 仍用 previous actual rate，不能把该拆分视为实际收到的现金流。

### Macro lookups 与 HPI matching

| Feature | Lookup month | 单位与 missing rule |
| --- | --- | --- |
| `c_orig_pmms` | `d_origination_month` | Percent；没有对应有效数值 → null |
| `c_pmms_lag1` | t−1 | Percent；同上 |
| `c_current_pmms` | t | Percent；同上 |
| `c_cpi_lag1` | t−1 | CPIAUCNS index level；同上 |
| `c_orig_hpi` | `d_origination_month` | ZHVI USD；按下述 geography pair 选择 |
| `c_hpi_lag1` | t−1 | ZHVI USD；同上 |
| `c_current_hpi` | t | ZHVI USD；沿用该行 pair 的 geography，当前值缺失时保持 null |
| `f_hpi_level` | Geography selection | String `state` 或 `national` |
| `f_hpi_region_id` | 所选 Zillow `region_id` | String；与该行的 HPI pair 一致 |

PMMS 使用 `MORTGAGE30US`，CPI 使用 `CPIAUCNS`；HPI 只使用 Zillow `ZHVI` 的 state / national rows。
Macro tables 的 month 必须为月初、lookup keys 唯一、数值有限，允许 null。
上游 PMMS weekly→monthly 的处理见 [housing macro data rules](../../housing_macro/data_rules.md)。

Zillow state 全名通过 `STATE_CODES` 映射到 Freddie 两位 state code；Zillow `RegionID` 不当作 Freddie MSA code。
当前不做 MSA/ZIP matching。**Origination 和 t−1 两个 state ZHVI 都为正时，整对使用 state；否则整对使用 national。**
这样 HPI ratio 来自同一个 geography，避免拿 national 的 base 和 state 的 current level 相除。
HPI 非正值先视为 null；national 仍缺失时保持 null，不做 interpolation、forward fill 或 future lookup。

Fallback 每个 loan-month 单独选择，因此同一贷款的 `c_orig_hpi` 可能随所选 geography 改变。
`c_current_hpi` 不参与 fallback 判断，也不触发第二次 fallback。`f_hpi_level` / `f_hpi_region_id` 用于追踪实际选择。

### Derived spreads 与 ratios

| Feature | Formula | 单位与 missing rule |
| --- | --- | --- |
| `c_sato` | `c_orig_rate - c_orig_pmms` | Origination spread，percentage points；任一输入缺失 → null |
| `c_incentive` | `c_prev_rate - c_pmms_lag1` | t−1 refinance incentive，percentage points；任一输入缺失 → null |
| `c_factor` | `c_prev_balance / c_orig_balance` | Ratio；分子缺失或分母缺失/≤0 → null |
| `c_hpi_growth` | `c_hpi_lag1 / c_orig_hpi - 1` | Ratio；分子缺失或分母缺失/≤0 → null |
| `c_updated_ltv` | `c_orig_ltv × c_factor × c_orig_hpi / c_hpi_lag1` | Percent；任一输入缺失或分母≤0 → null |

Rates 和 LTV 使用百分数，例如 `4.0` 表示 4%；SATO / incentive 使用百分点，例如 `4.0 - 3.5 = 0.5`。
Ratios 使用小数，例如 HPI growth 的 `0.10` 表示 10%。

SATO 衡量原贷款 rate 相对 origination PMMS 的 spread；incentive 使用 previous actual rate，能反映后续 rate modification。
`c_factor` 描述相对 original balance 的剩余余额比例。Updated LTV 假设房价按匹配 geography 的 ZHVI ratio 变化：
余额变化通过 factor 更新，property value 变化通过 HPI ratio 更新。这是 first-lien LTV 的近似，不更新 CLTV，
也不重建 individual-property appraisal、junior liens 或 exact contract amortization。

## 4. Status 与 quality

### 截至当前月的 modification 标记

`is_ever_modified` 为非空 Boolean：同一贷款按 reporting month 排序，首次观察到 `modification_flag=Y/P` 的当月及之后为 True；
之前为 False。后续 blank/null/其他编码不重置，月份断档也不重置；没有观察到 `Y/P` 时为 False。
它描述的是 **截至当前 row 已观察到 modification**，不使用未来信息给此前 rows 打标，也不保证披露前没有修改。
与只看连续上月的 `f_prev_modified` 不同，它跨断档保留累计状态，方便 downstream 排除修改后的 rows。
Summary 在每个 vintage 和总体记录 `ever_modified_rows`，不将这个标记视为 data-quality error。

### 合并 delinquency 与 exit information

`f_status` 是当前 row 的 descriptive state：有可识别 `zero_balance_code` 时使用 exit reason；
有未知非空 exit code 时为 `UNKNOWN_EXIT`；没有 exit code 时才使用 delinquency mapping。
`f_exit_reason` 只记录已识别的退出原因，没有或未知 exit code 时为 null。

| `current_loan_delinquency_status` | Delinquency state（`LOAN_STATUS_MAP`） |
| --- | --- |
| `00` | `CURRENT` |
| `01` | `DQ30` |
| `02` | `DQ60` |
| Numeric `03`–`99` | `DQ90_PLUS` |
| `RA` | `REO` |
| `XX`、blank/null、无法识别 | `UNKNOWN` |

Legacy 单位数字先补成两位再 lookup，例如 `1` → `01`。原始字段保留，所以以后仍可细分 DQ90 以上的 buckets。

| `zero_balance_code` | `f_exit_reason` / exit row 的 `f_status`（`EXIT_REASONS`） |
| --- | --- |
| `01` | `VOLUNTARY_PAYOFF`：包含提前还清和到期还清 |
| `02` | `THIRD_PARTY_SALE` |
| `03` | `SHORT_SALE_OR_CHARGE_OFF` |
| `09` | `REO_DISPOSITION` |
| `15` | `WHOLE_LOAN_SALE` |
| `16` | `REPERFORMING_SECURITIZATION` |
| `96` | `DEFECT`：credit event 前确认的 underwriting 或 major servicing defect |

Exit 表示退出 Freddie dataset 的跟踪；whole-loan sale / securitization 不意味着 borrower 已还清。
当前不根据 maturity 将 `01` 拆成 PREPAID / MATURED。
例如 exit code 为 `01`、delinquency code 为 `00`，得到 `f_status=VOLUNTARY_PAYOFF`，两列 raw codes 仍保留。

Event date 与 reporting month 不一致时，status 仍按当前 row 的编码生成；不移动 event、不改写以前的 status，
也不把 terminal state 自动延续到后面的 raw rows。`f_pre_status` 使用连续上一个月的最终 `f_status`。

### Quality flags

这些 Boolean fields 描述 source information，不自动决定 model eligibility；`True` 也不一定表示错误。

| Flag | `True` 的条件 |
| --- | --- |
| `is_consecutive_month` | 同一贷款存在连续上一个日历月的 row |
| `is_known_exit` | 当前 exit code 在 `EXIT_REASONS` 中，无论 event date 是否一致 |
| `is_unknown_exit_code` | 当前 exit code 非空但无法识别 |
| `is_missing_exit_month` | 当前 exit code 非空，但 `d_exit_month` 为 null |
| `is_event_month_mismatch` | 当前 exit code 非空、`d_exit_month` 非空，且与 reporting month 不同 |
| `is_zero_balance_without_exit` | 当前 actual UPB ≤0 且 exit code 为空；UPB 缺失时为 False |
| `is_post_exit` | 当前 reporting month 晚于整笔贷款最早的已识别退出月份 |

`is_post_exit` 的精确计算：对有已识别 exit code 的 rows，先取 reporting month 与 effective month 的较早者
（effective month 缺失时取 reporting month），再取整笔贷款的最早月份。没有已识别 exit 时为 False。
这是基于完整 loan history 的 retrospective flag；event/reporting mismatch 可能使报告 event 的那行也被标为 post-exit。
非法非空 event date 在 parsing 时直接报错，不会仅触发 `is_missing_exit_month`。

Summary 的 `quality_counts` 另外计数：previous balance 缺失/≤0、previous status 为 null 或 `UNKNOWN`、age 缺失或负数。
这些计数可以重叠，age=0 不属于 `invalid_age`；它们只用于检查，不导致删行。

## 5. 合成例子与 downstream 边界

### 完整 loan-month 的计算

假设 `first_payment_date=201502`，reporting month 为 2015-05，且存在 2015-04 的贷款记录。
清洗后 original balance 为 100,000 USD、original LTV 为 80、original rate 为 4.0、original term 为 360 months；
April actual balance 为 98,000 USD、actual rate 为 4.0，May actual balance 为 97,000 USD。
Origination PMMS 为 3.8，April PMMS 为 3.5；选中 geography 的 origination ZHVI 为 100,000 USD，April ZHVI 为 110,000 USD。

| 派生字段 | 结果 |
| --- | --- |
| `d_origination_month` | 2015-01-01 |
| `c_age` | 4 months |
| `c_prev_balance` / `c_prev_rate` | 98,000 USD / 4.0% |
| `c_balance` | 97,000 USD |
| `c_monthly_payment` | 约 477.4153 USD |
| `c_interest` | `98,000 × 4.0 / 1200 ≈ 326.6667` USD |
| `c_scheduled_principal` | `477.4153 − 326.6667 ≈ 150.7486` USD |
| `c_scheduled_balance` | `98,000 − 150.7486 ≈ 97,849.2514` USD |
| `c_sato` | `4.0 - 3.8 = 0.2` percentage points |
| `c_incentive` | `4.0 - 3.5 = 0.5` percentage points |
| `c_factor` | `98,000 / 100,000 = 0.98` |
| `c_hpi_growth` | `110,000 / 100,000 - 1 = 0.10` |
| `c_updated_ltv` | `80 × 0.98 × 100,000 / 110,000 ≈ 71.2727%` |

若 state 的 origination / April ZHVI 任一为空，整对改用 national，再计算 ratio；不会混用两种 geography。
若只缺 state 的 May ZHVI，pair 仍使用 state，`c_current_hpi` 留空。

### 月份断档

如果贷款有 March 和 May records，但没有 April record：

- May 的 `is_consecutive_month=False`；`c_prev_balance`、`c_prev_rate`、`f_prev_modified`、`f_pre_status` 为 null。
- April macro 若存在，May 的 `c_pmms_lag1`、`c_hpi_lag1`、`c_cpi_lag1` 仍可以有值。
- `c_incentive`、`c_factor`、`c_updated_ltv` 因缺少贷款 previous inputs 为 null；`c_hpi_growth` 仍可计算。
- `c_interest`、`c_scheduled_principal`、`c_scheduled_balance` 为 null；`c_monthly_payment` 不受断档影响。
- 若 February 已出现 `modification_flag=Y/P`，May 的 `is_ever_modified` 仍为 True，February 前的 rows 不被回标。
- 不补造 April row，也不使用 March balance 代替 April balance。

### 后续 model 自己决定什么

这个产物是 generic enriched panel，保留全部原始 observations，包括 terminal / post-exit rows、未知 status 和 feature nulls。
`feature_columns` 只是列清单，不是默认 predictor whitelist。这里不生成 `y_prepay`、`is_at_risk` 或 `is_model_eligible`。
Full/partial prepay target、risk set、missing-value treatment、predictor selection 和 train/test split 在 downstream 定义。

Reporting-month status、ending balance、current macro，以及使用完整历史的 flags，可能包含预测时尚未知的信息。
Macro `lag1` 仅表示 observation month 滞后一期；ZHVI/CPI revisions、PMMS monthly availability 等仍需在 model 阶段评估，
它不等于严格的 historical publication-time backtest。

日期、lag、reference payment、monthly split、累计 modification、fallback、sentinel、status、quality 和输出保留行为的合成验证，见
[test_freddie_features.py](../../../tests/data/test_freddie_features.py)。
