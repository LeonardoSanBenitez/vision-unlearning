Getting Started
===============

This guide will help you install Vision Unlearning and run your first unlearning experiment.

Installation
------------

Install the latest release from PyPI:

.. code-block:: sh

   pip install vision-unlearning

**Requirements:** Python 3.10 – 3.12.

For the optional platform integrations, install the relevant extras:

.. code-block:: sh

   pip install "vision-unlearning[huggingface]"   # Hugging Face Hub
   pip install "vision-unlearning[wandb]"          # Weights & Biases
   pip install "vision-unlearning[tensorboard]"    # TensorBoard

Or install all extras at once:

.. code-block:: sh

   pip install "vision-unlearning[all]"

Developer Installation
^^^^^^^^^^^^^^^^^^^^^^

Clone the repository and install in editable mode using `Poetry <https://python-poetry.org/>`_:

.. code-block:: sh

   git clone https://github.com/LeonardoSanBenitez/vision-unlearning.git
   cd vision-unlearning
   poetry install

Quick Start
-----------

The following examples show the minimal code needed to use each of the three main interfaces.
See :doc:`concepts` for a deeper explanation.

Unlearning a concept from a Text-to-Image model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vision_unlearning.unlearner.uce_sd_erase import UnlearnerUCE
   from vision_unlearning.datasets.local import LocalDataset

   # Define what to forget and what to keep
   dataset = LocalDataset(
       forget_prompts=["An image of a cat"],
       retain_prompts=["An image of a dog"],
   )

   # Configure and run the unlearner
   unlearner = UnlearnerUCE(
       model_name="CompVis/stable-diffusion-v1-4",
       dataset=dataset,
   )
   results = unlearner.train()

Evaluating an unlearned model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vision_unlearning.evaluator.text_to_image import EvaluatorTextToImage
   from vision_unlearning.metrics.fid import MetricFID

   evaluator = EvaluatorTextToImage(
       pipeline_unlearned=unlearned_pipeline,
       pipeline_original=original_pipeline,
       prompts_forget=["An image of a cat"],
       prompts_retain=["An image of a dog"],
       metrics=[MetricFID()],
   )
   report = evaluator.evaluate()
   print(report)

Using a standard dataset split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vision_unlearning.datasets.cifar import CIFARDataset

   dataset = CIFARDataset(
       split_mode="class",           # forget one entire class
       split_kwargs={"forget_class": 3},
   )
   forget_set = dataset.get_forget_split()
   retain_set = dataset.get_retain_split()

Next Steps
----------

* 📓 Follow the step-by-step :doc:`tutorials` to see complete worked examples.
* 💡 Read :doc:`concepts` to understand the design philosophy.
* 🔬 Explore the :doc:`testbeds` for reproducible research benchmarks.
* 📖 Browse the :doc:`autoapi/index` for the full API reference.
