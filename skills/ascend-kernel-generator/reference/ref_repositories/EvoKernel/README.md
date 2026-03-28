---
language:
- en
pretty_name: EvoKernel
viewer: false
task_categories:
- text-generation
tags:
- code
- pytorch
- cuda
- ascend
- npu
- kernel-synthesis
- text
---

# EvoKernel

This dataset accompanies the paper:

**Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis**

- Paper: https://arxiv.org/abs/2603.10846
- Project site: https://evokernel.zhuo.li

## What Is Included

This repository contains two kinds of artifacts:

1. **Exported kernel implementations**
   - `ops-attention-910b`: 58 curated Ascend 910B attention kernels
   - `ops-mhc-910b`: 15 curated Ascend 910B MHC kernels
   - `ops-kernelbench-910b`: 165 curated Ascend 910B KernelBench kernels
   - `ops-attention-cuda-ncu`: 79 CUDA attention kernels selected from iterative NCU-guided runs
   - `ops-kernelbench-cuda-ncu`: 250 CUDA KernelBench kernels selected from iterative NCU-guided runs
   - `ops-exports-bundle.zip`: bundled archive of the exported folders above

2. **PyTorch reference tasks**
   - `pytorch-references/Attention`: 79 PyTorch attention reference tasks
   - `pytorch-references/MHC`: 15 PyTorch MHC reference tasks
   - `pytorch-references/KernelBench`: PyTorch KernelBench references
     - `level1`: 100 tasks
     - `level2`: 100 tasks
     - `level3`: 50 tasks

## Format

Each exported kernel folder includes:

- implementation files
- a compact `result.json` with the selected iteration, correctness, and performance
- a root-level `manifest.json` summarizing all kernels in that collection

The `pytorch-references` folders contain the original PyTorch task definitions used to specify the kernel synthesis targets.

## Intended Use

This repository is intended for:

- studying kernel synthesis on Ascend 910B and CUDA backends
- reproducing task definitions used in EvoKernel experiments
- comparing generated kernels against PyTorch reference implementations
- benchmarking drafting and refinement pipelines on Attention, MHC, and KernelBench workloads

## Citation

If you use this dataset, please cite:

```bibtex
@article{zheng2026evokernel,
  title={Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis},
  author={Yujie Zheng and Zhuo Li and Shengtao Zhang and Hanjing Wang and Junjie Sheng and Jiaqian Wang and Junchi Yan and Weinan Zhang and Ying Wen and Bo Tang and Muning Wen},
  journal={arXiv preprint arXiv:2603.10846},
  year={2026}
}
```