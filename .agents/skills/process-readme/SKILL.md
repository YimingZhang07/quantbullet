---
name: process-readme
description: Write or restructure READMEs for runnable workflows, especially procs folders, as concise operating manuals describing execution order, purpose, commands, configuration and outputs. Move detailed logic and design reasoning into separate linked Markdown documents.
---

# Process README

Make the README a practical manual: the reader should quickly understand
what to run, in what order, with which inputs, and where results are written.
Keep detailed logic in separate documents linked from the manual.

## Check the current workflow

- Read the relevant entry points, example configs and existing documentation.
  Verify module names, CLI options, defaults, dependencies and output paths.
- Describe the current execution order, rather than the history of development.
  Identify prerequisites, independent branches and optional steps where relevant.
- Preserve the requested scope. Documentation work does not require changing
  pipeline code or running downloads, data builds or other external operations.

## README structure

Use the following sections as a starting point; omit or combine sections
that add no useful information:

1. **Purpose:** one or two sentences describing the workflow and its result.
2. **Setup:** required inputs, environment variables and config file links.
3. **Execution order:** a compact table with step, script/module, purpose and
   output. Give each step a short description and mark optional steps clearly.
4. **Commands:** copyable commands in execution order, with the working
   directory and project interpreter identified once.
5. **Common options and reruns:** only options and behavior needed to operate
   the workflow, such as refresh, subset selection and output replacement.
6. **Further reading:** links to the detailed rules and design documents.

Keep the manual short enough to scan. Do not repeat commands in multiple
sections or explain each implementation detail. A top-level README can
summarize dependencies and link to individual workflow manuals.
Use the requested language or follow the existing documentation language.

## Separate operations from logic

Keep information needed to run the workflow in the README: required paths,
parameters, supported choices, execution order, outputs and relevant failure
or overwrite behavior. Briefly explain a rule there only if the operator
needs it to choose a command or configuration.

Move the following into separate Markdown documents:

- Field definitions, schemas, status mappings and missing-value rules.
- Feature formulas, aggregation rules, date alignment and sampling algorithms.
- Model assumptions, methodological limitations and design tradeoffs.
- Implementation mechanics and reasoning about why an approach was selected.

Reuse an existing suitable document; otherwise choose a descriptive filename
such as `data_rules.md` or `design.md`. Do not create one file per small rule.
Move useful existing explanations without losing their meaning or changing
the rules. Add relative links from the README and a link back to the manual.
Keep detailed explanations in one place rather than duplicating them.
Remove obsolete migration instructions from the current manual when they
no longer describe the supported workflow.

## Final check

- Check that the reader can follow the order from inputs to outputs, including
  dependencies on other workflows.
- Verify commands and options against code, and check local document links.
- Use environment variables or generic placeholders for machine-specific
  paths; follow repository conventions for the Python interpreter.
- Confirm the README primarily answers "what to run and how to use it";
  detailed logic and "why this approach" belong in the linked documents.
