# Distance Correlation in PyTorch

Implementation of distance correlation (DC) and partial distance correlation (PDC) in PyTorch.
The code is written to be incorporated as a differentiable objective function.
This repo additionally includes an adaptation to differentiable Spearman correlation based on TorchSort.

## requirements

```
pytorch
numpy
torchsort
```

## installation

```
pip install git+https://github.com/hushon/distance-correlation-pytorch
```

## Example

Distance Correlation example

```python
from distance_correlation_pytorch import DistanceCorrelation

correlation = DistanceCorrelation(type="pearson") # pearson or spearman coefficient

batch_size = 128 
x = torch.rand(batch_size, 512)
y = torch.rand(batch_size, 2048)
corr = correlation(x, y)
```

Partial Distance Correlation example

```python
from distance_correlation_pytorch import PartialDistanceCorrelation

correlation = PartialDistanceCorrelation(type="pearson") # pearson or spearman coefficient

batch_size = 128 
x = torch.rand(batch_size, 512)
y = torch.rand(batch_size, 2048)
c = torch.rand(batch_size, 512) # conditioning variable for x
corr = correlation(x, y, condition_a=c)
```


## Bibliography

If you found this helpful with your project, please consider citing it.

```bibtex
@misc{hushon2022dc,
  author = {Hyounguk Shon},
  title = {Distance Correlation PyTorch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hushon/distance-correlation-pytorch}}
}
```