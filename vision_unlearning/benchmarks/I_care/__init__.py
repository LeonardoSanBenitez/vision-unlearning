"""I-CARE benchmark package.

This package contains all code for the I-CARE unlearning benchmark:
- configuration.py: domain constants, type aliases, GUI_TO_BACKEND mapping
- metadata.py: interference-per-pair / interference-per-entity helpers, InterferencePerEntity
- metrics.py: per-entity interference metric functions
- result_templates.py: ResultTemplate base class + all subclasses + registries
- utils.py: image encoding/decoding helpers, SHAP serialization, error classes

All public symbols are re-exported here so callers can do:
    from vision_unlearning.benchmarks.I_care import rt_name_to_class
    from vision_unlearning.benchmarks.I_care import InterferencePerEntity
    # etc.

Optional dependencies (seaborn, scikit-learn, shap) are used by the result templates
and the SHAP utilities.  They are not required for basic testbed / dataset work.
Install them via:  pip install vision-unlearning[testbed]
"""

from vision_unlearning.benchmarks.I_care.configuration import (
    domain_unlearning_algorithm,
    domain_task,
    domain_attribute,
    domain_entity,
    domain_model,
    domain_mp,
    domain_me,
    domain_s,
    domain_l,
    type_task,
    type_unlearning_algorithm,
    type_model,
    type_mp,
    type_me,
    type_s,
    type_l,
    GUI_TO_BACKEND,
    unlearning_algorithm_to_epochs,
    convert_params_from_gui_to_backend,
    s_to_direction,
    task_to_attributes_of_interest,
)

from vision_unlearning.benchmarks.I_care.metadata import (
    get_interference_per_pair_path,
    get_interference_per_pair,
    exists_interference_per_pair,
    save_interference_per_pair,
    get_interference_per_pair_inverse,
    get_interference_per_entity_path,
    get_interference_per_entity,
    save_interference_per_entity,
    InterferencePerEntity,
    choose_metric_column_interference_per_entity,
    get_embedding_output_path,
    get_embedding_hf_path,
)

from vision_unlearning.datasets.testbed import (
    get_metadata_filtered,
    get_metadata_filtered_path,
    get_target_overwrite,
    get_generated_dataset_file,
)

from vision_unlearning.benchmarks.I_care.metrics import (
    find_worst_interfered,
    metric_of_worst_interfered,
    is_worst_interfered_target,
    number_of_interfered_worse_than_target,
    number_of_interfered_worse_than_threshold,
    average_metric,
)

from vision_unlearning.benchmarks.I_care.utils import (
    _encode_image_file,
    _decode_image,
    explanation_to_dict,
    dict_to_explanation,
    InvalidAttributeTypeError,
    InsufficientSamplesError,
)

from vision_unlearning.benchmarks.I_care.embeddings import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    load_dino_model,
    embed_image_with_dino,
    embed_forgetting_session,
    embed_forgetting_session_batched,
)

from vision_unlearning.integrations.huggingface import (
    huggingface_dataset_file_exists,
    huggingface_dataset_file_download,
    huggingface_dataset_upload,
    huggingface_dataset_file_upload,
    huggingface_dataset_download,
)

from vision_unlearning.benchmarks.I_care.result_templates import (
    ResultTemplate,
    ResultTemplateMetricMetricAlignment,
    ResultTemplateMetricSimilarityAlignment,
    ResultTemplateMetricSimilarityAlignmentOne,
    ResultTemplateMetricSimilarityAlignmentMulti,
    ResultTemplateInterferenceBySimilarityRank,
    ResultTemplateMostSimilarMostInterferedGrid,
    ResultTemplateSignificantRelationshipNumerical,
    ResultTemplateSignificantRelationshipCategorical,
    ResultTemplateSignificantRelationshipCategoricalDirectional,
    ResultTemplateCountSignificantRelationship,
    ResultTemplateImplicitAssociationTest,
    ResultTemplateMinimumCutInterference,
    ResultTemplateUnlearningVisualSummary,
    ResultTemplateInterferenceVisualSummary,
    ResultTemplateMatrix,
    ResultTemplateInterferenceMatrix,
    ResultTemplateSimilarityMatrix,
    ResultTemplateMethodComparisonByMetricEntity,
    ResultTemplateEmbeddingUnlearningProfile,
    ResultTemplateEmbeddingForgettingEfficiency,
    rt_name_to_class,
    rt_name_to_params,
    jacc_metric_score,
    display_interesting_interferences,
    analyze_relationship_regression,
    analyze_relationship_category,
    analyze_relationship_numerical,
    analyze_relationship_categorical,
    analyze_correlation_between_pairwise_metrics,
    check_eval_results,
)

__all__ = [
    # configuration
    "domain_unlearning_algorithm",
    "domain_task",
    "domain_attribute",
    "domain_entity",
    "domain_model",
    "domain_mp",
    "domain_me",
    "domain_s",
    "domain_l",
    "type_task",
    "type_unlearning_algorithm",
    "type_model",
    "type_mp",
    "type_me",
    "type_s",
    "type_l",
    "GUI_TO_BACKEND",
    "unlearning_algorithm_to_epochs",
    "convert_params_from_gui_to_backend",
    "s_to_direction",
    "task_to_attributes_of_interest",
    # metadata
    "get_interference_per_pair_path",
    "get_interference_per_pair",
    "exists_interference_per_pair",
    "save_interference_per_pair",
    "get_interference_per_pair_inverse",
    "get_interference_per_entity_path",
    "get_interference_per_entity",
    "save_interference_per_entity",
    "InterferencePerEntity",
    "choose_metric_column_interference_per_entity",
    "get_embedding_output_path",
    "get_embedding_hf_path",
    # metrics
    "find_worst_interfered",
    "metric_of_worst_interfered",
    "is_worst_interfered_target",
    "number_of_interfered_worse_than_target",
    "number_of_interfered_worse_than_threshold",
    "average_metric",
    # utils
    "_encode_image_file",
    "_decode_image",
    "explanation_to_dict",
    "dict_to_explanation",
    "InvalidAttributeTypeError",
    "InsufficientSamplesError",
    # embeddings
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "load_dino_model",
    "embed_image_with_dino",
    "embed_forgetting_session",
    "embed_forgetting_session_batched",
    # result_templates
    "ResultTemplate",
    "ResultTemplateMetricMetricAlignment",
    "ResultTemplateMetricSimilarityAlignment",
    "ResultTemplateMetricSimilarityAlignmentOne",
    "ResultTemplateMetricSimilarityAlignmentMulti",
    "ResultTemplateInterferenceBySimilarityRank",
    "ResultTemplateMostSimilarMostInterferedGrid",
    "ResultTemplateSignificantRelationshipNumerical",
    "ResultTemplateSignificantRelationshipCategorical",
    "ResultTemplateSignificantRelationshipCategoricalDirectional",
    "ResultTemplateCountSignificantRelationship",
    "ResultTemplateImplicitAssociationTest",
    "ResultTemplateMinimumCutInterference",
    "ResultTemplateUnlearningVisualSummary",
    "ResultTemplateInterferenceVisualSummary",
    "ResultTemplateMatrix",
    "ResultTemplateInterferenceMatrix",
    "ResultTemplateSimilarityMatrix",
    "ResultTemplateMethodComparisonByMetricEntity",
    "ResultTemplateEmbeddingUnlearningProfile",
    "ResultTemplateEmbeddingForgettingEfficiency",
    "rt_name_to_class",
    "rt_name_to_params",
    "jacc_metric_score",
    "display_interesting_interferences",
    "analyze_relationship_regression",
    "analyze_relationship_category",
    "analyze_relationship_numerical",
    "analyze_relationship_categorical",
    "analyze_correlation_between_pairwise_metrics",
    "check_eval_results",
    # testbed helpers (re-exported for convenience)
    "get_metadata_filtered",
    "get_metadata_filtered_path",
    "get_target_overwrite",
    "get_generated_dataset_file",
    # huggingface integrations (re-exported for convenience and monkeypatching)
    "huggingface_dataset_file_exists",
    "huggingface_dataset_file_download",
    "huggingface_dataset_upload",
    "huggingface_dataset_file_upload",
    "huggingface_dataset_download",
]
