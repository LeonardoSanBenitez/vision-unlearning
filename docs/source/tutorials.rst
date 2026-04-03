Tutorials
=========

The tutorials below walk through complete, end-to-end unlearning experiments.
Each link opens an executed notebook stored in Google Drive (with full outputs).
The source notebooks (outputs cleared) are available in the ``tutorials/`` directory
of the `repository <https://github.com/LeonardoSanBenitez/vision-unlearning>`_.

Text-to-Image Unlearning with FADE
-----------------------------------

`FADE <https://arxiv.org/abs/2602.07058>`_ (Selective Forgetting via Sparse LoRA &
Self-Distillation) replaces the representation of a forgotten concept with a target concept
while preserving the rest of the model's knowledge.

.. list-table::
   :widths: 60 40
   :header-rows: 1

   * - Tutorial
     - Links
   * - Replace *George W. Bush* → *Tony Blair* using FADE
     - `▶ Open notebook (FADE) <https://colab.research.google.com/drive/1ZJG9By4_u1Vqy_SYelxfzUUImRzayRYw?usp=sharing>`_
   * - Replace *George W. Bush* → *Tony Blair* using FADE sparse-per-module
     - `▶ Open notebook (FADE sparse-per-module) <https://colab.research.google.com/drive/1luM3kAoaBLoTwcsDcW3KO_SIiuIbwmWY?usp=sharing>`_
   * - Replace *George W. Bush* → *Tony Blair* using FADE sparse-per-weight
     - `▶ Open notebook (FADE sparse-per-weight) <https://colab.research.google.com/drive/1ry5xXOPMuVm_LA_4Uyk27Aqe52L607kO?usp=sharing>`_

Text-to-Image Unlearning with UCE
----------------------------------

`UCE <https://github.com/rohitgandikota/unified-concept-editing>`_ (Unified Concept Editing)
edits model weights to erase concepts without requiring additional training data.

.. list-table::
   :widths: 60 40
   :header-rows: 1

   * - Tutorial
     - Links
   * - Forget *cat* using UCE (with hyperparameter tuning)
     - `▶ Open notebook (UCE) <https://drive.google.com/file/d/1OZtNkntOj-dVpo-T1kQdPMK7TMYX3ctf/view?usp=sharing>`_

Text-to-Image Unlearning with Munba
-------------------------------------

Munba is a gradient-based method that selectively erases concepts from diffusion models.

.. list-table::
   :widths: 60 40
   :header-rows: 1

   * - Tutorial
     - Links
   * - Forget *church* using Munba
     - `▶ Open notebook (Munba) <https://colab.research.google.com/drive/1eyjrNMcYi0PK37U0ZLcwydy153yiJUJ9?usp=sharing>`_

Notes for Developers
--------------------

Every time a relevant modification is made to the codebase, the affected tutorials should be
re-run and the executed notebook saved to Google Drive. Before committing, clear the notebook
output to avoid burdening the repository.
