# OPUS MSMV Sampling

This directory vendors the official OPUS-V1 four-level multi-scale,
multi-view sampling CUDA kernel. Build it from this directory:

```bash
python setup.py build_ext --inplace
```

The encoder imports `msmv_sampling` from this package. It uses the CUDA kernel
when available and retains the official tensor layout in its PyTorch fallback
for CPU-only tests.
