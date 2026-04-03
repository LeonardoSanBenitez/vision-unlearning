.. vision-unlearning documentation master file

Vision Unlearning
=================

.. image:: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/mypy.yml/badge.svg?branch=dev&job=mypy
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/mypy.yml
   :alt: Mypy

.. image:: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pycodestyle.yml/badge.svg?branch=dev&job=pycodestyle
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pycodestyle.yml
   :alt: Pycodestyle

.. image:: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pytest.yml/badge.svg?branch=dev&job=pytest
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/pytest.yml
   :alt: Pytest

.. image:: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/publish.yml/badge.svg
   :target: https://github.com/LeonardoSanBenitez/vision-unlearning/actions/workflows/publish.yml
   :alt: Publish Package to PyPI

**Vision Unlearning** provides a standard interface for unlearning algorithms, datasets, metrics,
and evaluation methodologies commonly used in Machine Unlearning for vision-related tasks
such as image classification and image generation.

It bridges the gap between research/theory and engineering/practice, making it easier to apply
machine unlearning techniques effectively.

.. grid:: 2

    .. grid-item-card:: 🚀 Getting Started
        :link: getting_started
        :link-type: doc

        Install the library and run your first unlearning experiment in minutes.

    .. grid-item-card:: 💡 Core Concepts
        :link: concepts
        :link-type: doc

        Learn about the three main interfaces: ``Metric``, ``Unlearner``, and ``Dataset``.

.. grid:: 2

    .. grid-item-card:: 📓 Tutorials
        :link: tutorials
        :link-type: doc

        Step-by-step notebooks for FADE, UCE, and Munba algorithms.

    .. grid-item-card:: 🔬 Testbeds & Benchmarks
        :link: testbeds
        :link-type: doc

        Standardized meta-benchmarks for reproducible research.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   getting_started
   concepts
   tutorials
   testbeds
   benchmarks

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   autoapi/index
