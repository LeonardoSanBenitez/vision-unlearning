Tutorials
=========

The tutorials below walk through complete end-to-end workflows using
Vision Unlearning.  Each notebook is stored on Google Drive / Colab with full
output included so you can read the results without running anything locally.

The source notebooks (outputs cleared) live in ``tutorials/`` in the repository.

Unlearning with FADE
--------------------

`Replace George W. Bush by Tony Blair using FADE
<https://colab.research.google.com/drive/1ZJG9By4_u1Vqy_SYelxfzUUImRzayRYw?usp=sharing>`_

A standard person-replacement task using the FADE algorithm.  Covers dataset
setup, algorithm configuration, and metric evaluation.

`Replace George W. Bush by Tony Blair using FADE sparse-per-module
<https://colab.research.google.com/drive/1luM3kAoaBLoTwcsDcW3KO_SIiuIbwmWY?usp=sharing>`_

Same task with the sparse-per-module variant of FADE, which applies weight
pruning per transformer module instead of globally.

`Replace George W. Bush by Tony Blair using FADE sparse-per-weight
<https://colab.research.google.com/drive/1ry5xXOPMuVm_LA_4Uyk27Aqe52L607kO?usp=sharing>`_

Sparse-per-weight variant: the sparsity mask is applied at individual weight
level for finer-grained control.

Unlearning with UCE
-------------------

`Forget "cat" using UCE (with hyperparameter tuning)
<https://drive.google.com/file/d/1OZtNkntOj-dVpo-T1kQdPMK7TMYX3ctf/view?usp=sharing>`_

A scene/class-removal task using the UCE (Unified Concept Erasure) algorithm.
Demonstrates hyperparameter search and evaluation with retain-performance metrics.

Unlearning with Munba
---------------------

`Forget "church" using Munba
<https://colab.research.google.com/drive/1eyjrNMcYi0PK37U0ZLcwydy153yiJUJ9?usp=sharing>`_

Scene-removal task using the Munba algorithm.  Covers both forgetting quality
and side-effect metrics on retained concepts.
