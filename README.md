# quantbullet

`quantbullet` is a toolkit designed for streamlined quantitative analysis in finance. The goals for this package are:

1. To provide a practical set of tools for prototyping quantitative research ideas.
2. To integrate and test contemporary research findings, primarily from academic sources, ensuring they're actionable.

While I initially developed this package for my own needs, I intend to maintain it consistently. If it assists others in their endeavors, I consider that a success.

## Installation

```bash
$ pip install quantbullet
```

## Usage

1. Statistical Jump Models. See [this notebook](./docs/research/jump_model_prod.ipynb) for an example. Statistical jump models are a type of regime-switching model that applies clustering algorithms to temporal financial data, explicitly penalizing jumps between different financial regimes to capture true persistence in underlying regime-switching processes.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`quantbullet` was created by Yiming Zhang. It is licensed under the terms of the MIT license.

## Credits

This project developement is generously supported by JetBrains softwares with their Open Source development license.

<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" alt="JetBrains Logo (Main) logo." width=200>

`quantbullet` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter). [Python Packages](https://py-pkgs.org/) is an excellent resource for learning how to create and publish Python packages.
