# quantbullet

`quantbullet` is a toolkit for swift quantitative analysis for finance. My vision for this package is to

1. Provide a set of tools for fast prototyping of quantitative research ideas
2. Implement and test the latest research ideas (mostly will come from academic papers) in a production-ready manner

The package is mainly for my personal use, so I will commit to maintaining it for a long time. However, I will be happy if it can help others as well.

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

`quantbullet` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter). [Python Packages](https://py-pkgs.org/) is an excellent resource for learning how to create and publish Python packages.
