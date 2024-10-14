# AHGCOND
This is the official PyTorch implementation of AHGCOND.

## Environment Requirement

Hardware environment: Intel(R) Xeon(R) Silver 4208 CPU, a Quadro RTX 6000 24GB GPU, and 128GB of RAM.

Software environment:Python 3.9.12, Pytorch 1.13.0, and CUDA 11.2.0.

## Run the Code

For hypergraph condensation on Cora/Pubmed/DBLP/Walmart/Yelp

```bash
python main.py
```

For hypergraph condensation on MAG-PM

```bash
python main_large.py
```