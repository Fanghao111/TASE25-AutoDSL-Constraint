# [IEEE T-ASE] Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation

[![T-ASE](https://img.shields.io/badge/IEEE%20T--ASE-2025-blue)](https://doi.org/10.1109/TASE.2025.3596540)

This is the official resource repository for IEEE T-ASE [paper](https://doi.org/10.1109/TASE.2025.3596540) Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation.

## Overview

Advanced Planning and Scheduling (APS) systems have become indispensable for modern manufacturing operations, enabling optimized resource allocation and production efficiency in increasingly complex and dynamic environments. While algorithms for solving abstracted scheduling problems have been extensively investigated, the critical prerequisite of specifying manufacturing requirements into formal constraints remains manual and labor-intensive. Although recent advances of generative models, particularly Large Language Models (LLMs), show promise in automating constraint specification from heterogeneous raw manufacturing data, their direct application faces challenges due to natural language ambiguity, non-deterministic outputs, and limited domain-specific knowledge. This paper presents a constraint-centric architecture that regulates LLMs to perform reliable automated constraint specification for production scheduling. The architecture defines a hierarchical structural space organized across three levels, implemented through domain-specific representation to ensure precision and reliability while maintaining flexibility. Furthermore, an automated production scenario adaptation algorithm is designed and deployed to efficiently customize the architecture for specific manufacturing configurations. Experimental results demonstrate that the proposed approach successfully balances the generative capabilities of LLMs with the reliability requirements of manufacturing systems, significantly outperforming pure LLM-based approaches in constraint specification tasks.

***Note to Practitioners:***

This paper presents a practical solution for automating the conversion of raw manufacturing information into job scheduling specifications, addressing a common challenge in implementing APS systems. The proposed architecture can process diverse manufacturing documentation formats, from semi-structured route sheets to natural language instructions, while ensuring reliability through domain-specific representations. Manufacturing practitioners can use this system to reduce the manual effort in specifying digitalized constraints for production, particularly beneficial for facilities with frequent requirement changes or small-batch, multi-variety production. The system’s ability to automatically adapt to different manufacturing scenarios makes it accessible without requiring extensive programming expertise, offering a practical balance between automation and accuracy in production planning.


## Repository Structure

- `main.py`: entry script for preprocessing, AutoDSL construction, baselines, DSL pipeline execution, and evaluation.
- `src/`: core implementation of preprocessing, DSL design, experiments, evaluation, and prompt templates used by the pipeline.
- `data/`: released scheduling inputs and intermediate structured data.
- `utils/`: shared utility functions used across the pipeline.

## Quick Start

### Setup

```bash
cd archive
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Minimal Run Path

On the released data in this archive, the following deterministic stages can be executed directly:

```bash
python main.py --mode preprocess
python main.py --mode autodsl_operation
python main.py --mode autodsl_production
python main.py --mode groundtruth
```

These commands write derived artifacts under `data/` and `outputs/`.

After the corresponding method outputs are available, evaluation can be run with:

```bash
python main.py --mode evaluation --evaluation_type CPE
python main.py --mode evaluation --evaluation_type CAE
python main.py --mode evaluation --evaluation_type CSE-1
python main.py --mode evaluation --evaluation_type CSE-2
python main.py --mode evaluation --evaluation_type SGE
```

### Reproducibility Notes

- LLM-based stages additionally require an `OPENAI_API_KEY`.
- `baseline`, `baseline_2`, and `dsl_pipeline` depend on intermediate artifacts under `outputs/` when resumed from partially generated results.
- `autodsl_operation` loads `allenai/scibert_scivocab_uncased`, which may trigger a model download on first use.

## Citation

```
@article{shi2025automated,
  title={Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation},
  author={Shi, Yu-Zhe and Xu, Qiao and Li, Yanjia and Liu, Mingchen and Qu, Huamin and Ruan, Lecheng and Wang, Qining},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2025},
  doi={10.1109/TASE.2025.3596540},
  publisher={IEEE}
}
```

## License

Please refer to the LICENSE file for specific licensing terms.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
