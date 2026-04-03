Core Concepts
=============

Vision Unlearning is built around **three main abstractions** that cover every step of a
machine unlearning workflow:

.. image:: ../images/UML.png
   :alt: UML diagram of the main interfaces
   :align: center

|

The Three Interfaces
--------------------

Metric
^^^^^^

A ``Metric`` evaluates how well a model has forgotten the target concept while preserving
general capability.  Every metric is a `Pydantic <https://docs.pydantic.dev/>`_ model,
which means its configuration is type-checked and serialisable.

**Built-in metrics include:**

+-------------------------------+--------------------------------------+
| Class                         | Description                          |
+===============================+======================================+
| ``MetricFID``                 | Fréchet Inception Distance           |
+-------------------------------+--------------------------------------+
| ``MetricImageTextSimilarity`` | CLIP-based image–text similarity     |
+-------------------------------+--------------------------------------+
| ``MetricPaintingStyle``       | Painting-style classifier            |
+-------------------------------+--------------------------------------+
| ``MetricRace``                | Face-based demographic attribute     |
+-------------------------------+--------------------------------------+

**Extending Metric:**

.. code-block:: python

   from vision_unlearning.metrics.base import Metric

   class MyMetric(Metric):
       threshold: float = 0.5

       def compute(self, images, **kwargs) -> float:
           ...

Unlearner
^^^^^^^^^

An ``Unlearner`` encapsulates a complete unlearning algorithm.  Its only required method is
``train()``, which modifies the model in-place and returns a list of ``EvalResult`` objects.

**Built-in unlearners:**

+------------------+---------------------------------------------------------------+
| Class            | Algorithm                                                     |
+==================+===============================================================+
| ``UnlearnerFADE``| FADE – Selective Forgetting via Sparse LoRA & Self-Distillation |
+------------------+---------------------------------------------------------------+
| ``UnlearnerUCE`` | UCE – Unified Concept Editing for Stable Diffusion            |
+------------------+---------------------------------------------------------------+
| ``UnlearnerLoRA``| Generic LoRA-based fine-tuning / unlearning                   |
+------------------+---------------------------------------------------------------+

**Extending Unlearner:**

.. code-block:: python

   from vision_unlearning.unlearner.base import Unlearner
   from typing import List
   from vision_unlearning.evaluator.base import EvalResult

   class MyUnlearner(Unlearner):
       learning_rate: float = 1e-4

       def train(self) -> List[EvalResult]:
           # Modify self.model here
           ...
           return []

Dataset
^^^^^^^

An ``UnlearnDataset`` wraps any dataset and manages the split between the **forget set**
(the data the model should no longer remember) and the **retain set** (everything else).

Three split strategies are provided out of the box:

* **class** – forget all samples belonging to one or more classes.
* **random** – forget a randomly sampled subset.
* **temporal** – forget samples recorded after a certain date.

**Built-in datasets:**

+--------------------+--------------------------------------+
| Class              | Dataset                              |
+====================+======================================+
| ``CIFARDataset``   | CIFAR-10 / CIFAR-100                 |
+--------------------+--------------------------------------+
| ``ImageNette``     | ImageNette (10-class ImageNet subset)|
+--------------------+--------------------------------------+
| ``COCODataset``    | COCO                                 |
+--------------------+--------------------------------------+
| ``LocalDataset``   | Any folder of images on disk         |
+--------------------+--------------------------------------+
| ``TestbedDataset`` | Vision-Unlearning testbed datasets   |
+--------------------+--------------------------------------+

**Extending UnlearnDataset:**

.. code-block:: python

   from vision_unlearning.datasets.base import UnlearnDataset, UnlearnDatasetSplitMode

   class MyDataset(UnlearnDataset):
       split_mode: UnlearnDatasetSplitMode = UnlearnDatasetSplitMode.class_

       def load(self):
           ...

       def get_forget_split(self):
           ...

       def get_retain_split(self):
           ...

Evaluator
---------

The ``Evaluator`` classes combine a trained (or unlearned) model with a set of ``Metric``
objects and a ``Dataset`` to produce a structured evaluation report.

+-----------------------------+------------------------------------------+
| Class                       | Use case                                 |
+=============================+==========================================+
| ``EvaluatorTextToImage``    | Text-to-image diffusion models           |
+-----------------------------+------------------------------------------+
| ``EvaluatorClassUnlearning``| Image-classification models              |
+-----------------------------+------------------------------------------+

Platform Integrations
---------------------

Vision Unlearning ships optional, lazily-loaded integrations for common ML platforms.
Import them only when you need them; the core library has no hard dependency on them.

* **Hugging Face Hub** – push/pull models and datasets (``vision_unlearning.integrations.huggingface``)
* **Weights & Biases** – log metrics and artefacts (``vision_unlearning.integrations.wandb``)
* **TensorBoard** – scalar logging (``vision_unlearning.integrations.tensorboard``)
* **Local storage** – save results to disk (``vision_unlearning.integrations.local``)

Design Principles
-----------------

Vision Unlearning is designed to be:

* **Easy to use** – sensible defaults, Pydantic-validated configuration.
* **Easy to extend** – every interface is an abstract base class; subclass and override.
* **Architecture-agnostic** – works with any PyTorch model.
* **Application-agnostic** – supports classification, generation, and beyond.
