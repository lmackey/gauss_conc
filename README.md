# Efficient Concentration with Gaussian Approximation

Replication code for [Efficient Concentration with Gaussian Approximation](https://arxiv.org/pdf/2208.09922.pdf):
- [`efficient.py`](efficient.py) implements the efficient known-variance tail bounds and quantile bounds
- [`ebe.py`](ebe.py) implements the efficient empirical Berry-Esseen bound
- [`plot_quantile_bounds.ipynb`](plot_quantile_bounds.ipynb) reproduces the quantile bound, efficient quantile bound, and empirical quantile bound experiments
- [`monte_carlo_confidence.ipynb`](monte_carlo_confidence.ipynb) reproduces the Monte Carlo confidence intervals for numerical integration experiment

```bibtex
@article{
  austern2022efficient,
  title={Efficient Concentration with Gaussian Approximation},
  author={Morgane Austern and Lester Mackey},
  journal={arXiv preprint arXiv:2208.09922},
  year={2022}
}
```

## Setup instructions

This code has been tested with Python 3.12.

```bash
conda create -n gauss python=3.12
conda activate gauss
pip install "numpy<2" scipy qmcpy matplotlib seaborn jupyterlab
pip install git+https://github.com/gostevehoward/confseq.git
pip install ttictoc sympy 
```
