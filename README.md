# Visual Computing Algorithms

Selected implementations extracted from personal COMP0169 solution regions (2025–26), packaged as small importable modules. This is a selection of algorithms, not the full teaching notebook or assignment.

| Module | Implementations |
|---|---|
| `regression.py` | Bias-augmented least squares, linear fitting, quadratic features |
| `image_filters.py` | Normalised Gaussian kernel and colour-image filtering |
| `neural_components.py` | Denoising CNN, MSE loss, feature Gram/mean/variance and normalisation |

## Quick start

Install `requirements.txt`, then run:

```python
import numpy as np
from regression import line_fit
x = np.arange(8, dtype=float)[:, None]
y = 1 + 2 * x[:, 0]
weights, mse = line_fit(x, y)
print(weights, mse)
```

Install `requirements-neural.txt` only if using the PyTorch components. No datasets, pretrained networks or trained checkpoints are included. Neural code is provided as implementation components, not a trained denoising product.

The original course scaffold, teaching utilities, assessment prompts, cloud-drive integration and upload credentials are omitted. Selected function/class interfaces and return wiring are retained so the implementation bodies can be used independently. Course context: COMP0169, taught by Niloy J. Mitra and Tobias Ritschel.

## Publication scope

This is a curated portfolio of coursework implementations from the author's local working files. Course questions, marking rubrics, slides, reports, student identifiers, notebook outputs, input datasets, trained weights and commercial models are not included. Original private archives remain separate.

Supply your own appropriately licensed inputs where required. Existing algorithm limitations are preserved; publication is not a claim of a new benchmark or a complete reproduction of the original assessment. Dependencies retain their own licences. No blanket licence is added to third-party material.
