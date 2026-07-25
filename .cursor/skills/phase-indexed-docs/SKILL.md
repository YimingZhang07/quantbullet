---
name: phase-indexed-docs
description: Create concise phase-indexed README / index docs for staged development work. Use when organizing docs folders by phase, sequencing plan/demo/tie-out files, or when the user wants a Chinese-English index showing each phase, generated files, and purpose.
---

# Phase-Indexed Docs

Use this skill when a docs folder is growing around a staged feature / refactor，而用户想要的是一个清晰的阶段索引，而不是一篇很长的 guide。

## Core Pattern

Name this documentation style **phase-indexed docs**：按照 development phase 组织 README / index。每个 phase 下面列出这一阶段产生的 files，以及每个 file 的 purpose。

Prefer this shape:

````markdown
# <Feature> Docs

一句话说明这个 folder 是做什么的。

## Phase 1: <Phase Name>

- `<plan-file>.md`: 一句话说明这个 phase 的 plan / design boundary。
- `<demo-or-validation>.py`: 一句话说明这个 script demo 或 validate 什么。

## Phase 2: <Phase Name>

- `<next-plan>.md`: 一句话说明下一步 design。
- `<next-script>.py`: 一句话说明 runnable output 或 tie-out 结果。

## Running Scripts

```powershell
.\.venv\Scripts\python.exe docs\<folder>\<script>.py
```

Generated outputs stay local，并且应该被 git ignore。
````

## Ordering Rules

- 按 development sequence 排，不按 filename 或 file type 排。
- 每个 phase 里优先按这个顺序列：plan/design doc → demo script → tie-out/validation script → supporting notes。
- 每个 file 只写一到两句短说明，focus on purpose，不展开实现细节。
- 不要移动 existing paths，除非用户明确要求 restructure。
- Shared run commands 放在最后，不要在每个 file 下面重复。
- Generated artifacts 只在最后统一说明一次。

## What To Avoid

- 如果 phase order 本身就是 reading order，不要再单独写 "Recommended Reading Order"。
- README 是 index，不是 tutorial；不要写成长篇说明。
- 当用户的 mental model 是 staged development 时，不要按 "markdown files" / "Python files" 这种文件类型拆分。
- 不要过度解释 implementation details；link file，然后说明 purpose 即可。
