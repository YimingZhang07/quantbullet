---
name: general-knowledge
description: 把问题当成一个独立的、project-independent 的新问题来回答，依据 general domain knowledge 和 external sources，而不是这个 repo 里的 code、docs、notes 或之前的 conclusions。Use when the user invokes $general-knowledge，或者想要 exploratory、conceptual、learning-oriented 的回答、不希望被当前 project anchor 住，例如：不要参考 project、当成一个新问题、用通用知识回答、fresh perspective、clean-slate answer。
---

# General Knowledge

把问题当成在 repo 之外被问到的。用户要的是基于 general domain knowledge 和
authoritative sources 的回答，而不是被这个 project 的 code、docs 或 past
decisions 塑造出来的回答。Project 里的内容可能 incomplete、有错，或者只适用于
当时的情况；依赖它，就会把这些局限带进答案。

## Sources to use

- 自己的 domain knowledge：standard definitions、textbook methods、
  established practice、well-known results。
- External sources：知识可能 outdated、incomplete 或 contested，或者问题依赖
  具体事实（library behavior、recent research、regulations、market conventions、
  data definitions）时，去 search。优先 primary / authoritative sources：papers、
  textbooks、official docs、standards、data provider 自己的 documentation。
  用到的要 cite。
- 用户在问题里直接写的内容。这是 question 的一部分，不算 project context。

## Sources to set aside

- 不要为了回答去 read / search / list repo files：source code、tests、docs、
  notes、READMEs、configs、notebooks、generated outputs 都不看。
- 已经在 context 里的 project knowledge 也不用，包括 loaded instructions、
  memory、这段对话前面的 findings，以及 repo 的 naming 和 design choices。
  当它们不存在。
- 不要围绕 project 的 modules、functions、classes 或 terminology 组织答案，
  也不要按"适合现有 code"调整 recommendation。用 standard terminology 和
  neutral、self-contained 的 examples。
- 不要把 project 的做法当成 standard 或 correct answer。

## Answering

- 从问题本身出发：先 define concepts，再讲 mechanism、standard approaches，
  以及它们为什么 work。
- 做法有分歧时，列出 main alternatives，说明各自的 assumptions、trade-offs 和
  typical failure modes。
- 区分 established consensus、自己的 judgment、仍然 uncertain / debated 的点。
  不确定就直说，不要拿 project-specific details 填空。
- Learning-oriented 的问题，给出值得继续读的 references。
- 跟随用户的 language 和要求的 detail level。

## Boundaries

- 这个 skill 只改变 knowledge 从哪里来，不改变 operating rules。讨论转成实际
  工作时，repo 关于 environment、git、public hygiene 的规则照常适用。
- 用户在问题里主动贴出的 file、snippet 或 project detail：只用贴出的部分，并用
  general knowledge 去 evaluate，不当作 authoritative。不要为了补 context 去
  打开其他 files。
- 没有 project specifics 就答不好时，直说，并问用户要不要看 project，而不是
  silently 去读。
- 除非用户要求，不要拿答案和 project 的 implementation 做 comparison。对比明显
  有帮助时，可以在结尾用一句话 offer。
- 同一段讨论里的 follow-up questions 保持这个 mode，直到用户要求把 project
  带回来，或者转到需要在 repo 里实际工作的 task。
