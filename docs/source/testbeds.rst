Testbeds
========

Vision Unlearning ships *testbeds* — meta-benchmarks that provide a standardised starting
point for designing new assessments, benchmarks, or unlearning interventions.

All heavy media (images, generated data, pre-trained models) is published on Zenodo:

   https://doi.org/10.5281/zenodo.18649818

.. note::
   We are still uploading content and intend to finish in the coming months.
   Reach out to `Leonardo Benitez <https://github.com/LeonardoSanBenitez>`_ if you are
   interested in the data before it is fully available.

Methodology
-----------

Each testbed is produced by the following pipeline:

.. image:: ../images/testbeds_generation_pipeline.png
   :alt: Testbed generation pipeline
   :align: center

|

1. Choose **3 unlearning methods** representative of the main algorithm categories.
2. Choose **3 tasks** representative of unlearning applications.
3. Choose **2 or more attributes of interest** (visual, unambiguous, not controversial).
4. Enrich all entities with those attributes.
5. Restrict to **100 entities**, intersectionally balanced across the attributes.
6. Sequentially equalise hyperparameters across unlearning methods.
7. Train unlearned models and generate images.

Standardised Assets
^^^^^^^^^^^^^^^^^^^

Every task provides the following files/assets:

``metadata_filtered`` (``metadata_{task}_2_enriched_filtered.json``)
    List of dicts, one per entity.  Fields include ``name``, ``index``, ``is_unlearned``,
    ``dataset_n_original``, and task-specific attributes.

``similarity_clip`` (``similarity_clip_{task}.json``)
    100×100 pairwise CLIP-score similarity matrix between entity names.

``similarity_attr`` (``similarity_attr_{task}.json``)
    100×100 pairwise attribute-similarity matrix (Jaccard + scaled absolute difference).

Data splits
    Separate folders for forget and retain images:

    * ``{dataset_base_path}/{target}/train_forget``
    * ``{dataset_base_path}/{target}/train_retain``

Unlearned models
    900 Stable Diffusion v1-4 models, one per ``(task, target, method, epoch)`` combination.
    Stored at ``models/{task}_{target}_{method}_{num_train_epochs:03d}``.

Generated images
    360 000 images produced by the unlearned models.
    Stored at ``datasets/generated_{task}_{target}_{method}_{num_train_epochs:03d}``.

Basic Testbeds
--------------

These tasks cover common, well-understood visual domains.

Breeds
^^^^^^

*Unlearning a dog breed recognised by the FCI.*

* **Main image dataset:** ``taras_breeds``
* **Attribute datasets:** ``akc``, ``pawsomeauthority``
* **Entities:** 100 dog breeds

Key attributes: ``description``, ``temperament``, ``popularity``, ``min_height``, ``group``
(Sporting, Hound, Working, Terrier, Toy, Non-Sporting, Herding, Miscellaneous, FSS).

Scenes
^^^^^^

*Unlearning a scene — a holistic, semantically coherent environment.*

* **Main image dataset:** ``SUN``
* **Attribute datasets:** ``pantheon``
* **Entities:** 100 scenes

Key attributes: ``socializing``, ``natural``, ``open_area``, ``exercise``.

People
^^^^^^

*Unlearning one famous person.*

* **Main image dataset:** ``lfw``
* **Attribute datasets:** ``pantheon``
* **Entities:** 100 public figures balanced by ``occupation_simplified`` and ``hpi_bin``

Key attributes: ``race``, ``gender``, ``birthyear``, ``occupation``, ``hpi``
(Historical Popularity Index).

Objects
^^^^^^^

*Still under development.*

Applied Testbeds
----------------

These tasks target higher-stakes, domain-specific applications.

Biomedical
^^^^^^^^^^
*Still under development.*

Autonomous Vehicles
^^^^^^^^^^^^^^^^^^^
*Still under development.*

Art Adaptation
^^^^^^^^^^^^^^
*Still under development.*

Known Works Using These Testbeds
---------------------------------

* `Attribute-Based Interpretation (ABI) <https://github.com/sohamnilvaze/attribute_based_interpretation>`_ —
  an ongoing framework for Explainable AI.
* **I-CARE** — Interference in Concept Adaptation and geneRative Erasure.
  See :doc:`benchmarks`.
* `FADE: Selective Forgetting via Sparse LoRA and Self-Distillation <https://arxiv.org/abs/2602.07058>`_
