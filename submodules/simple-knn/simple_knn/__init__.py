# Simple KNN CUDA extension
import torch  # noqa: F401 - required to load shared libraries
from simple_knn._C import distCUDA2

