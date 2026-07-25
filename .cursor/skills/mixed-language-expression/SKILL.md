---
name: mixed-language-expression
description: Use efficient Chinese-English mixed expression for notes, explanations, technical writing, and AI-assisted documentation. Use when the user asks for 中英文混杂, 中英混合, mixed Chinese-English writing, expression efficiency, note polishing, or technical notes where Chinese should connect the logic and English should preserve key nouns, verbs, code terms, and technical terminology.
---

# Mixed Language Expression

Use this skill to write or edit notes in a clear, efficient Chinese-English mixed style.

## Goal

提高表达效率，而不是为了 mixed language 而 mixed language.

Use Chinese for semantic flow:

- 背景说明
- 因果关系
- 判断和 takeaway
- 语义串联
- 对复杂点的解释

Use English for high-signal terms:

- key nouns: `training data`, `model`, `cache`, `source of truth`, `feature`, `target`
- technical verbs: `train`, `fit`, `write`, `read`, `resolve`, `cache`, `dump`, `merge`
- code artifacts: file paths, scripts, functions, config names, env vars
- finance / quant / engineering terms where English is more precise or standard

## Core Rules

1. Keep the sentence readable. 不要每个词都切换语言。
2. Prefer English when Chinese translation is awkward, lossy, or less precise.
3. Prefer Chinese when explaining logic, implications, and why something matters.
4. Keep technical terms stable. Pick one term and reuse it.
5. Preserve exact code terms, paths, function names, CLI flags, and config names.
6. Avoid unnecessary full-Chinese translations of standard technical terms.
7. Do not over-polish into formal English; the target is efficient working notes.

## Good Style

Use this kind of phrasing:

```text
Training data 不在 repo 里。Repo is code only.
```

```text
`run.py` 调 `prep_transition.py`，后者 writes train parquet 到 `R:\...\_train\...`。
```

```text
R side 用 `model\models.R` resolve `paths$infile`，然后 `model\train.R` reads parquet and fits GAM。
```

```text
这个 cache 是 transition training 的 upstream input，不是最终 model-ready data。
```

## Avoid

Avoid awkward all-Chinese technical translation:

```text
训练数据由准备转换脚本产生，然后由模型训练脚本读取并拟合广义加性模型。
```

Prefer:

```text
`prep_transition.py` writes transition train parquet; `model\train.R` reads it and fits the GAM。
```

Avoid excessive English when Chinese would connect the idea better:

```text
The producer script writes the train parquet and the consumer script resolves the infile path and performs model fitting.
```

Prefer:

```text
最短链路是：producer writes train parquet；consumer resolves `paths$infile` and fits model。
```

## Note Editing Preference

When editing notes:

- Keep content concise.
- Add only durable points worth remembering.
- Prefer short sections with direct answers.
- Use code blocks for paths and script chains.
- Do not turn working notes into a comprehensive runbook unless the user asks.
