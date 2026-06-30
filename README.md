# Vision Unlearning

<!-- ![CI](https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/tests.yml/badge.svg) -->

<!-- Seperate batches for 3 tests-->
![Mypy](https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/mypy.yml/badge.svg?branch=dev&job=mypy)
![Pycodestyle](https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pycodestyle.yml/badge.svg?branch=dev&job=pycodestyle)
![Pytest](https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pytest.yml/badge.svg?branch=dev&job=pytest)
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/LeonardoSanBenitez/vision-unlearning/dev/coverage.json)
[![Release](https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/vision-unlearning.svg)](https://pypi.org/project/vision-unlearning/)
[![Python](https://img.shields.io/pypi/pyversions/vision-unlearning.svg)](https://pypi.org/project/vision-unlearning/)
[![License](https://img.shields.io/github/license/LeonardoSanBenitez/vision-unlearning)](LICENSE)


[Documentation](https://vision-unlearning.readthedocs.io/)

## Installation

```sh
pip install vision-unlearning
```

Compatible with python 3.10 to 3.12.

## What is Vision Unlearning?

Vision Unlearning provides a standard interface for unlearning algorithms, datasets, metrics, and evaluation methodologies commonly used in Machine Unlearning for vision-related tasks, such as image classification and image generation.

It bridges the gap between research/theory and engineering/practice, making it easier to apply machine unlearning techniques effectively.

Vision Unlearning is designed to be:
- Easy to use
- Easy to extend
- Architecture-agnostic
- Application-agnostic

## Who is it for?

### Researchers
For Machine Unlearning researchers, Vision Unlearning helps with:
- Using the same data splits as other works, including the correct segmentation of forget-retain data and generating data with the same prompts.
- Choosing the appropriate metrics for each task.
- Configuring evaluation setups in a standardized manner.

### Practitioners
For practitioners, Vision Unlearning provides:
- Easy access to state-of-the-art unlearning algorithms.
- A standardized interface to experiment with different algorithms.

# Tutorials

* [Replace _George W. Bush_ by _Tony Blair_ using FADE](https://colab.research.google.com/drive/1ZJG9By4_u1Vqy_SYelxfzUUImRzayRYw?usp=sharing)
* [Replace _George W. Bush_ by _Tony Blair_ using FADE sparse-per-module](https://colab.research.google.com/drive/1luM3kAoaBLoTwcsDcW3KO_SIiuIbwmWY?usp=sharing)
* [Replace _George W. Bush_ by _Tony Blair_ using FADE sparse-per-weigth](https://colab.research.google.com/drive/1ry5xXOPMuVm_LA_4Uyk27Aqe52L607kO?usp=sharing)
* [Forget _cat_ using UCE (with hyperparam tunning)](https://drive.google.com/file/d/1OZtNkntOj-dVpo-T1kQdPMK7TMYX3ctf/view?usp=sharing)
* [Forget _church_ using Munba](https://colab.research.google.com/drive/1eyjrNMcYi0PK37U0ZLcwydy153yiJUJ9?usp=sharing)

The source code for these tutorials is in `tutorials/`, but their outputs were cleaned to avoid burdening the repo.
The links above contain Google Drive stored executions with the full outputs.

> For developers: every time there is a relevant modification in the codebase, please run the affected tutorials, save the notebook to Drive, clear the output before commiting.


# Benchmarks

Vision-Unlearning provides easy access to evaluation benchmarks.

Across all benchmarks, a standardized set of files/media/metadata/content is provided following the structure described in the Appendix 2 of I-CARE paper. As such, they are compatible with [Forgety](https://github.com/LeonardoSanBenitez/forgety). 


## I-CARE

Even though the feasibility demonstration of the I-CARE methodology is not per se a full-fledged benchmark, it is exposed by Vision-Unlearning as one. Three independent tasks are analyzed (forgetting people, scenes, and dog breeds) across three unlearning methods  from the state-of-the-art (UCE, SPARE, MUNBa). Data, models, metrics and computed results are provided its [HuggingFace repository](huggingface.co/datasets/LeonardoBenitez/VisionUnlearningEvaluationTestbeds). See code details in `vision_unlearning/benchmarks/I_care`.


This work analyzes how unlearning one entity (the emitter) affects the performance on other closely-related entities (the receivers). Each task is defined by carefully selecting a diverse and representative set of entities (concepts that will undergo unlearning). Each entity is annotated with relevant attributes, and separately unlearned using different unlearning methods. Each unlearned model is then used to generate images for all entities, allowing fine-grained analysis of the effects caused by the unlearning process. Last but not least, all entities contain the same amount of images and were carefully selected so as to be balanced across at least 2 attributes.


## Unlearn Canvas
TODO... still under construction...

For more information on the benchmark itself, please refer to their repository https://github.com/OPTML-Group/UnlearnCanvas.


Data, models, metrics and computed results used by Vision-Unlearning are [or will be] provided [this HuggingFace repository](https://huggingface.co/datasets/LeonardoBenitez/u-care).
See code details in `vision_unlearning/benchmarks/u_care`.

## Holistic Unlearning Benchmark (HUB)
TODO... still under construction...

For more information on the benchmark itself, please refer to their paper: https://arxiv.org/abs/2410.05664

Data, models, metrics and computed results used by Vision-Unlearning are [or will be] provided [this HuggingFace repository](https://huggingface.co/datasets/LeonardoBenitez/hub-care).
See code details in `vision_unlearning/benchmarks/hub_care`.

# Forgety
[Forgety](https://github.com/LeonardoSanBenitez/forgety) is a related project built on top of this library, providing a web-based graphical UI.



# Main Interfaces

Vision Unlearning standardizes the following components:

- **Metric**: Evaluates a model (e.g., FID, CLIP Score, MIA, NudeNet, etc.).
- **Unlearner**: Encapsulates the unlearning algorithm.
- **Dataset**: Encapsulates the dataset, including data splitting.

Additionally, common tasks and evaluation setups are provided as example notebooks. Several platform integrations, such as Hugging Face and Weights & Biases, are also included.

![uml](docs/images/UML.png)

# Citation

We don't have yet a paper specifically about the library. Instead, please cite the first paper that was build using vision-unlearning:

```
@misc{mola2026,
      title={SPARE: Self-distillation for PARameter-Efficient Removal}, 
      author={Natnael Mola and Leonardo S. B. Pereira and Carolina R. Kelsch and Luis H. Arribas and Juan C. S. M. Avedillo},
      year={2026},
      eprint={2602.07058},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.07058}, 
}
```



