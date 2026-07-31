from typing import Dict, List, Literal, Set

from pydantic import BaseModel

from vision_unlearning.benchmarks.configuration import (
    MetricWithDirectionSpec,
    UnlearningAlgorithmSpec,
)

type_task = Literal["unlearncanvas"]
type_model = Literal["sd_style50"]
type_domain = Literal["style", "object"]
type_unlearning_algorithm = Literal[
    "ca", "ediff", "esd", "fmn", "salun", "seot", "shs", "spm", "uce",
]
type_mp = Literal[
    "accuracy", "accuracy_diff", "target_probability", "target_probability_diff",
]
type_me = Literal[
    "Unlearning accuracy",
    "In domain retain accuracy",
    "Cross domain retain accuracy",
    "Frechet inception distance",
    "Runtime seconds",
    "Peak memory bytes",
]

# Copy the four lists VERBATIM from their const.py, order included. The classifier was
# trained with label index == position in theme_available / class_available, so argmax
# index i means STYLE_ENTITIES[i] (or OBJECT_ENTITIES[i]); a reordered list mislabels
# every prediction while still looking plausible.
STYLE_ENTITIES: List[str] = ["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
"Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
"Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
"Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
"Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
"Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
"Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
"Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]   # 51, == their theme_available
OBJECT_ENTITIES: List[str] = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers", "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers", "Trees", "Waterfalls"]  # 20, == their class_available
ENTITIES: List[str] = STYLE_ENTITIES + OBJECT_ENTITIES               # 71

# Fail loudly at import if any list drifts.
assert len(STYLE_ENTITIES) == 51
assert len(OBJECT_ENTITIES) == 20
assert len(ENTITIES) == 71
assert set(STYLE_ENTITIES).isdisjoint(OBJECT_ENTITIES)

# --- "can be unlearned" logic (the one entity that cannot) ---
NON_UNLEARNABLE_ENTITIES: Set[str] = {"Seed_Images"}


def is_unlearnable(entity: str) -> bool:
    """An entity can be unlearned (used as an emitter) iff a model can be produced with it
    forgotten. Every painting style and every object class qualifies. `Seed_Images` is the
    un-stylised source condition: a class the style classifier recognises and a receiver in
    every answer set, but never a concept a model is unlearned on, so it cannot be an emitter.
    """
    return entity not in NON_UNLEARNABLE_ENTITIES


UNLEARNABLE_ENTITIES: List[str] = [e for e in ENTITIES if is_unlearnable(e)]  # 70
assert len(UNLEARNABLE_ENTITIES) == 70


def entity_domain(entity: str) -> type_domain:
    """The domain attribute of an entity. `Seed_Images` is a style class, so it is `style`."""
    return "style" if entity in STYLE_ENTITIES else "object"


ANSWER_SET_SEEDS: List[int] = [188, 288, 588, 688, 888]
U_CARE_REMOTE_REPOSITORY_NAME = "LeonardoBenitez/u-care"


def model_segment(model: type_model) -> str:
    """Filename/folder segment for the base model. Mirrors I_care.configuration.model_segment."""
    return f"_{model}"


def answer_set_prompt(theme: str, object_class: str) -> str:
    """The prompt the unlearned-model samplers use. It does NOT map Seed_Images to 'Photo',
    although fine-tuning did — see the release inconsistencies above."""
    return f"A {object_class} image in {theme.replace('_', ' ')} style."


MP_REGISTRY: Dict[type_mp, MetricWithDirectionSpec] = {
    "accuracy": MetricWithDirectionSpec(
        name="accuracy", name_pretty="Recognition Accuracy", direction="↑"),
    "accuracy_diff": MetricWithDirectionSpec(
        name="accuracy_diff", name_pretty="Delta Recognition Accuracy", direction="↑"),
    "target_probability": MetricWithDirectionSpec(
        name="target_probability", name_pretty="Target Probability", direction="↑"),
    "target_probability_diff": MetricWithDirectionSpec(
        name="target_probability_diff", name_pretty="Delta Target Probability", direction="↑"),
}

ALGORITHM_REGISTRY: Dict[type_unlearning_algorithm, UnlearningAlgorithmSpec] = {
    "ca": UnlearningAlgorithmSpec(name="ca", name_pretty="CA"),
    "ediff": UnlearningAlgorithmSpec(name="ediff", name_pretty="EDiff"),
    "esd": UnlearningAlgorithmSpec(name="esd", name_pretty="ESD"),
    "fmn": UnlearningAlgorithmSpec(name="fmn", name_pretty="FMN"),
    "salun": UnlearningAlgorithmSpec(name="salun", name_pretty="SalUn"),
    "seot": UnlearningAlgorithmSpec(name="seot", name_pretty="SEOT"),
    "shs": UnlearningAlgorithmSpec(name="shs", name_pretty="SHS"),
    "spm": UnlearningAlgorithmSpec(name="spm", name_pretty="SPM"),
    "uce": UnlearningAlgorithmSpec(name="uce", name_pretty="UCE"),
}

_UNLEARNING_ALGORITHM_DISPLAY_ORDER: List[type_unlearning_algorithm] = [
    "ca", "ediff", "esd", "fmn", "salun", "seot", "shs", "spm", "uce",
]


class UnlearningConfiguration(BaseModel):
    """The published hyperparameters for one (method, domain). There is no epoch dimension:
    UnlearnCanvas reproduces one published hyperparameter configuration per method, so the
    configuration itself is recorded rather than an epoch count."""
    erase_scale: float
    lamb: float
    guided_concept: str


UNLEARNING_CONFIGURATION: Dict[type_unlearning_algorithm, Dict[type_domain, UnlearningConfiguration]] = {
    "uce": {
        # Read from their UCE method README. "A Elephant image" is their exact string, article
        # error included — keep it verbatim, it produced their numbers.
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="An image in Photo style"),
        "object": UnlearningConfiguration(erase_scale=0.01, lamb=10.0, guided_concept="A Elephant image"),
    },
    # ... the remaining methods, read from each method's own README
    "ca": {
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="Painting of olive trees in style of Van Gogh"),
        "object": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="What a cute Grumpy cat"),
    },
    "ediff": {},
    "esd": {
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="Thomas Kinkade inspired depiction of a peaceful park"),
        "object": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="car"),
    }, 
    "fmn": {
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="Claude Monet style"),
        "object": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="superman"),
    },
    "salun": {
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="Unearn \"Van Gogh\""),
                "object": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="Unlearn \"Dog\""),
    },
    "seot": {
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="A dog in Van Gogh Style"),
                "object": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="A man with a beard wearing glasses and a hat in blue shirt"),
    },
    "shs": {},
    "spm": {
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="A vase of vibrant flowers, in style of Van Gogh's still lifes"),
        "object": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="Dog"),
    }

}