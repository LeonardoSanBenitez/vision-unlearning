Benchmarks
==========

Vision Unlearning provides wrappers for two benchmarks that evaluate unlearning algorithms
in a standardised way.

Unlearn Canvas
--------------

`UnlearnCanvas <https://github.com/OPTML-Group/UnlearnCanvas>`_ is an external benchmark
for assessing machine unlearning methods on diffusion models.  Vision Unlearning exposes a
thin interface so you can run it directly from our library:

.. code-block:: python

   from vision_unlearning.benchmarks.unlearn_canvas import UnlearnCanvas

   benchmark = UnlearnCanvas(
       unlearner=my_unlearner,
       dataset=my_dataset,
   )
   results = benchmark.run()

For full details on the benchmark methodology, refer to the
`UnlearnCanvas repository <https://github.com/OPTML-Group/UnlearnCanvas>`_.

I-CARE
------

**Interference in Concept Adaptation and geneRative Erasure** analyses how unlearning one
entity (the *emitter*) affects performance on other closely-related entities (the *receivers*).
It leverages the :doc:`testbeds` provided by Vision Unlearning.

.. note::
   Most of the I-CARE code is currently in a private repository.  We intend to open-source
   it in the coming months.  Reach out to
   `Leonardo Benitez <https://github.com/LeonardoSanBenitez>`_ if you are interested.

Result Files
^^^^^^^^^^^^

``interference_per_pair``
    ``Dict[str, Dict[str, float]]`` — ``MetricInterferencePerEntityPair`` values averaged
    across all seeds.  One file per unlearning session.

    *Path:* ``datasets/interferences_caused_by_{task}_{index}_{method}_{num_train_epochs}.json``

``interference_per_entity``
    ``List[Dict[str, Any]]`` — ``MetricInterferencePerEntity`` joined with all attributes
    from ``metadata_filtered``.  One file per task, across all methods.

    *Path:* ``interference_per_entity_{task}.json``
