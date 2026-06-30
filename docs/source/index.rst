Vision Unlearning
=================

.. image:: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/mypy.yml/badge.svg?branch=dev
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/mypy.yml

.. image:: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pycodestyle.yml/badge.svg?branch=dev
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pycodestyle.yml

.. image:: https://img.shields.io/pypi/v/vision-unlearning.svg
   :target: https://pypi.org/project/vision-unlearning/

.. image:: https://img.shields.io/pypi/pyversions/vision-unlearning.svg
   :target: https://pypi.org/project/vision-unlearning/

.. image:: https://img.shields.io/github/license/LeonardoSanBenitez/vision-unlearning
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/blob/main/LICENSE

|

A standard interface for unlearning algorithms, datasets, metrics, and evaluation
in vision-related machine learning tasks.

.. code-block:: bash

   pip install vision-unlearning

Compatible with Python 3.10 to 3.12.

----

What is Vision Unlearning?
--------------------------

Machine unlearning is the problem of removing the influence of specific training
data from a trained model — without retraining from scratch.  Vision Unlearning
provides a standard, architecture-agnostic interface for the core building blocks:

- **Unlearning algorithms** — FADE, UCE, Munba, and others
- **Datasets** — standardised forget/retain splits for CIFAR-10/100, CelebA, and more
- **Metrics** — quantitative measures of forgetting quality, retain performance, and side effects
- **Evaluator** — a unified pipeline for running an algorithm and measuring all relevant metrics
- **I-CARE benchmark** — a comprehensive evaluation testbed for comparing methods across tasks and entities

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   autoapi/vision_unlearning/unlearner/index
   autoapi/vision_unlearning/datasets/index
   autoapi/vision_unlearning/metrics/index
   autoapi/vision_unlearning/evaluator/index
   autoapi/vision_unlearning/integrations/index
   autoapi/vision_unlearning/utils/index
   autoapi/vision_unlearning/benchmarks/index
