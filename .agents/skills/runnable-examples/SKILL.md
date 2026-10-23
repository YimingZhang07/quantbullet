---
name: runnable-examples
description: 为一个 library 功能写或 review 用户视角的 runnable examples（放在 examples/ 下）：fake data、self-contained、arguments explicit，每个用法一个 test method，可在 VS Code Test panel 用 run button 单独运行，plot / CSV / PDF 写到可配置的 output folder。Use when the user invokes $runnable-examples，或者要求写 examples、用法示例、usage demo、"从 user 的角度展示怎么用"、review 已有的 examples。
---

# Runnable Examples

Examples 是给 user 看的用法说明，不是 test suite。读者打开一个 test method，应该能马上看懂：
用什么数据 → 怎么调这个功能（每个 argument 是什么意思）→ 得到什么输出、去哪里看。
正式的 correctness tests 仍然放在 `tests/`。

参考实现：[examples/diagnostics/test_two_way_examples.py](../../../examples/diagnostics/test_two_way_examples.py)。

## 位置和运行

- 一个功能一个文件：`examples/<area>/test_<feature>_examples.py`，`<area>` 对应 package 的区域（例如 `diagnostics`、`reporting`）。
- 文件名以 `test_` 开头，pytest 才会收集。
- 运行方式：
  - VS Code Test panel 的 run button，可以跑一个 method，也可以跑整个 class。
    这需要 `python.testing.pytestArgs` 包含 `"examples"`（`.vscode/` 是本地的、被 git ignore，只能提醒用户自己加）。
  - Command line：`.venv/Scripts/python.exe -m pytest examples/<area>/test_<feature>_examples.py -k <name>`。
- `pyproject.toml` 的 `testpaths = ["tests"]` 让不带参数的 `pytest` 不收集 examples，所以 examples 不会拖慢 test suite。不要改这一点。
- `addopts = "-s"` 已经开着，`print` 的 tables 直接显示在 output 里。

## 结构：一个 `unittest.TestCase`，setup 只放公约数

- 整个文件一个 `class Test<Feature>Examples(unittest.TestCase)`。**不要**自己写 parent class 再继承，也不要搞 mixin。
  用户觉得这种写法 hacky、不直观。
- `setUpClass` 只放**每个** example 都需要的东西（最大公约数），通常是：
  - fake data；
  - 被测对象的共同 constructor settings，写成带注释的 dict（例如 `cls.columns = dict(...)`）。
- **不要**在 setup 里预先算只有部分 examples 才用的东西（某个 method 的结果、fit、table）。
  否则用户会以为某个用法依赖这些 setup。
- 用 `setUpClass`，不用 `setUp`：
  - 只点一个 method 运行时两者一样；
  - 全部运行时 `setUp` 会把数据重复生成 N 次；
  - 只有 test 会改动共享对象时，才需要 `setUp` 的隔离。共享对象应当是 read-only 的（例如 Polars 的 `filter` / `with_columns` 都返回新 frame）。
- 每个 test method 是一个**完整的用法**：自己构造对象（例如 `diag = Thing(self.data, **self.columns)`），调用要展示的 method，打印或写出结果。
  需要变体时在同一行里改，例如 `{**self.columns, "pred": "other_column"}`。
- 多个 examples 共用的取值（bins、knots、date ranges 等）写成 module-level constants，和 `OUTPUT_DIR` 放在一起。它们是取值，不是 setup。
- `fake_*` data generator、`show()`、`output_path()` 这类 helper 写成 module-level functions，不放进 class。

## Fake data

- 只用 synthetic data，在文件里用一个有 docstring 的 function 生成（例如 `fake_loan_months(n_rows, seed)`），固定 seed。
- 不读本地数据，不依赖 `tests/` 里的 helpers。整个文件 self-contained，只 import library 的 public API 和常规依赖。
- 数据里**埋一个已知的效应**（例如模型漏掉的一个 interaction），并在 module docstring 里写清楚。
  这样 output 有可解释的答案，example 可以把 fitted 结果和 truth 并排打印。
- 数据量要够大，让 pattern 清楚、不被噪声干扰；又要够小，单个 example 几秒内跑完。先跑一遍看 output 是否"一眼看得懂"，噪声大就加数据。

## Arguments explicit

- 用 keyword arguments 调用，主要选项即使等于默认值也写出来。
- 第一次出现某个 argument 时加简短行内注释：它做什么、有什么 alternative（例如 `slices="x",  # one panel per x bin; "z" gives one per z bin`）。
- 后面的 examples 可以写得紧凑一些，但仍然保留 keyword 名称。
- 只用 public API（例如 `from quantbullet.diagnostics import ...`），不碰 private helpers。

## Outputs

- `OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "<feature>"`，放在文件顶部并注释"Change it to write somewhere else"。
- 只在写文件的 test 里通过 `output_path(name)` 创建 folder；只打印的 test 不碰它。
- Figures 存 PNG（`dpi=150, bbox_inches="tight"`），tables 存 CSV，需要成品时写 PDF（例如 `PdfPages`，或 repo 的 report 工具）。写完打印路径。
- `matplotlib.use("Agg")`：不弹窗口，只写文件。
- `.gitignore` 里已有 `examples/**/output/`，generated files 不进 git。

## Assertions

- 少量 sanity assertions，证明这个用法 works，并检查埋入的已知效应：方向对、在几个 SE 以内、文件写出来了。
- 用 `self.assert...`。不要把 examples 写成穷举的 tests。

## Module docstring

依次说明：

1. 这些 examples 展示什么功能；
2. `setUpClass` 里放了什么；
3. 怎么跑一个 example（VS Code run button，以及一条 `pytest -k` 命令）；
4. outputs 写到哪里、git ignore；
5. fake data 里埋了什么已知效应。

## Skeleton

```python
"""<Feature> on fake data: one runnable example per test method. ..."""
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files; no plot windows

from quantbullet.<area> import <Feature>  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "<feature>"   # change to write elsewhere
SHARED_SETTING = ...                                                    # values several examples reuse


def fake_data(n_rows: int, seed: int):
    """What the columns are, and the known effect hidden in them."""


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


class Test<Feature>Examples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = fake_data(n_rows=..., seed=0)       # every example starts from this ...
        cls.settings = dict(                           # ... and these constructor settings
            option_a="...",   # what it does; alternatives
        )

    def test_basic_usage(self):
        """One usage: build, call, show the output."""
        thing = <Feature>(self.data, **self.settings)
        result = thing.method(arg=SHARED_SETTING, other_arg=...)  # explicit keywords
        print(result.table())
        self.assertTrue(...)

    def test_plot(self):
        thing = <Feature>(self.data, **self.settings)
        plot = thing.method(arg=SHARED_SETTING).plot(option=...,  # comment
                                                     title="...")
        path = output_path("plot.png")
        plot.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"written: {path}")
```

## Final check

- 整个文件跑一遍、单独跑一个 method，都 pass；记下大致耗时。
- `pytest --collect-only -q tests examples` 能列出每个 method，也就是 VS Code 能看到的那些；不带参数的 `pytest` 收集不到 examples。
- 打开生成的 PNG / PDF 实际看一眼：标签有没有重叠、刻度是否可读、pattern 是否和 docstring 描述一致。
- 检查有没有重复的 setup 代码、setup 里有没有只被部分 examples 用到的东西、有没有 personal paths。generated outputs 被 git ignore。
