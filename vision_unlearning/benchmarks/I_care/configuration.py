# Configuration for the specific setup described in the paper.
# Defines domain types, metadata, and task-to-attribute mappings.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, get_args

from pydantic import BaseModel

# domain_unlearning_algorithm (pretty names) is derived from ALGORITHM_REGISTRY, defined
# further down alongside the other declarative registries (Mp/S/L/UnlearningAlgorithm).
domain_task = ["Breeds", "Scenes", "People"]
domain_attribute = {
    "Breeds": [
        #'dataset_n_original',
        'group',
        'section',
        'country',
        #'name_akc',
        #'description',
        'temperament',
        'popularity',
        'min_height',
        'max_height',
        'min_weight',
        'max_weight',
        'min_expectancy',
        'max_expectancy',
        'group_akc',
        'grooming_frequency_value',
        'grooming_frequency_category',
        'shedding_value',
        'shedding_category',
        'energy_level_value',
        'energy_level_category',
        'trainability_value',
        'trainability_category',
        'demeanor_value',
        'demeanor_category',
        #'URL',
        #'Pronunciation',
        #'Other Names',
        #'Nickname',
        'Origin',
        'group_pawsome',
        'Size',
        'Male Height Min (in)',
        'Male Height Max (in)',
        'Male Height Min (cm)',
        'Male Height Max (cm)',
        'Female Height Min (in)',
        'Female Height Max (in)',
        'Female Height Min (cm)',
        'Female Height Max (cm)',
        'Male Weight Min (lbs)',
        'Male Weight Max (lbs)',
        'Male Weight Min (kg)',
        'Male Weight Max (kg)',
        'Female Weight Min (lbs)',
        'Female Weight Max (lbs)',
        'Female Weight Min (kg)',
        'Female Weight Max (kg)',
        'Coat Length',
        'Coat Type',
        'Double Coat',
        'Hypoallergenic',
        'Affection Rating',
        'Playfulness Rating',
        'Protectiveness Rating',
        'Territoriality Rating',
        'Prey Drive Rating',
        'Barking Rating',
        'Good with Children Rating',
        'Good with Adults Rating',
        'Good with Dogs Rating',
        'Good with Pets Rating',
        'Good with Strangers Rating',
        'Sociability Rating',
        'Sensitivity Rating',
        'Separation Anxiety Rating',
        'Energy Rating',
        'Intelligence Rating',
        'Mental Stimulation Rating',
        'Obedience Rating',
        'Trainability Rating',
        'Stubbornness Rating',
        'Attention Span',
        'Shedding Rating',
        'Grooming Rating',
        'Drooling Rating',
        'Lifespan Min',
        'Lifespan Max',
        'Health Rating',
        'Dental Issues Rating',
        'Ear Issues Rating',
        'Eye Issues Rating',
        'Owner Experience Rating',
        'First Time Owner',
        'Apartment Living Rating',
        'grooming_frequency_category_binary',
    ],
    "Scenes": [
        'sailing/ boating',
        'driving',
        'biking',
        'transporting things or people',
        'sunbathing',
        'vacationing/ touring',
        'hiking',
        'climbing',
        'camping',
        'reading',
        'studying/ learning',
        'teaching/ training',
        'research',
        'diving',
        'swimming',
        'bathing',
        'eating',
        'cleaning',
        'socializing',
        'congregating',
        'waiting in line/ queuing',
        'competing',
        'sports',
        'exercise',
        'playing',
        'gaming',
        'spectating/ being in an audience',
        'farming',
        'constructing/ building',
        'shopping',
        'medical activity',
        'working',
        'using tools',
        'digging',
        'conducting business',
        'praying',
        'fencing',
        'railing',
        'wire',
        'railroad',
        'trees',
        'grass',
        'vegetation',
        'shrubbery',
        'foliage',
        'leaves',
        'flowers',
        'asphalt',
        'pavement',
        'shingles',
        'carpet',
        'brick',
        'tiles',
        'concrete',
        'metal',
        'paper',
        'wood (not part of a tree)',
        'vinyl/ linoleum',
        'rubber/ plastic',
        'cloth',
        'sand',
        'rock/stone',
        'dirt/soil',
        'marble',
        'glass',
        'waves/ surf',
        'ocean',
        'running water',
        'still water',
        'ice',
        'snow',
        'clouds',
        'smoke',
        'fire',
        'natural light',
        'direct sun/sunny',
        'electric/indoor lighting',
        'aged/ worn',
        'glossy',
        'matte',
        'sterile',
        'moist/ damp',
        'dry',
        'dirty',
        'rusty',
        'warm',
        'cold',
        'natural',
        'man-made',
        'open area',
        'semi-enclosed area',
        'enclosed area',
        'far-away horizon',
        'no horizon',
        'rugged scene',
        'mostly vertical components',
        'mostly horizontal components',
        'symmetrical',
        'cluttered space',
        'scary',
        'soothing',
        'stressful',
        #'dataset_n_original',
    ],
    "People": [
        #'dataset_n_original',
        'birthyear',
        'gender',
        'occupation',
        'bplace_country',
        'hpi',
        'race',
        'occupation_simplified',
        'hpi_bin',
    ],
}
domain_entity = {
    "Breeds": [
        'dogo argentino',
        'griffon bruxellois dog',
        'griffon belge dog',
        'norwegian elkhound black dog',
        'norwegian elkhound grey dog',
        'tibetan terrier dog',
        'leonberger dog',
        'pekingese dog',
        'flat coated retriever dog',
        'bosnian and herzegovinian - croatian shepherd dog',
        'border terrier dog',
        'basenji dog',
        'american staffordshire terrier dog',
        'bouvier des ardennes dog',
        'bouvier des flandres dog',
        'nova scotia duck tolling retriever dog',
        'coton de tulear dog',
        'staffordshire bull terrier dog',
        'rottweiler dog',
        'chinese crested dog',
        'giant schnauzer dog',
        'chow chow dog',
        'great swiss mountain dog',
        'old english sheepdog',
        'lhasa apso dog',
        'miniature pinscher dog',
        'cairn terrier dog',
        'welsh corgi (cardigan) dog',
        'dogue de bordeaux',
        'shar pei dog',
        'bull terrier dog',
        'airedale terrier dog',
        'samoyed dog',
        'alaskan malamute dog',
        'scottish terrier dog',
        'australian cattle dog',
        'continental toy spaniel dog',
        'irish soft coated wheaten terrier dog',
        'english cocker spaniel dog',
        'bullmastiff dog',
        'portuguese water dog',
        'bulldog',
        'st. bernard dog',
        'akita dog',
        'bichon frise dog',
        'chesapeake bay retriever dog',
        'shiba dog',
        'belgian shepherd dog',
        'west highland white terrier dog',
        'newfoundland dog',
        'french bulldog',
        'maltese dog',
        'border collie dog',
        'miniature american shepherd dog',
        'chihuahua dog',
        'golden retriever dog',
        'pug dog',
        'english springer spaniel dog',
        'shetland sheepdog',
        'canarian warren hound dog',
        'bernese mountain dog',
        'boston terrier dog',
        'german shepherd dog',
        'norwegian lundehund dog',
        'miniature schnauzer dog',
        'berger de beauce dog',
        'cesky terrier dog',
        'finnish spitz dog',
        'finnish lapponian dog',
        "cirneco dell'etna dog",
        'pyrenean sheepdog - smooth faced',
        'sussex spaniel dog',
        'king charles spaniel dog',
        'cavalier king charles spaniel dog',
        'canaan dog',
        'skye terrier dog',
        'dandie dinmont terrier dog',
        'irish glen of imaal terrier dog',
        'komondor dog',
        'polish lowland sheepdog',
        'australian shepherd dog',
        'american water spaniel dog',
        'sealyham terrier dog',
        'kuvasz dog',
        'curly coated retriever dog',
        'puli dog',
        'irish water spaniel dog',
        'spanish water dog',
        'nederlandse kooikerhondje dog',
        'field spaniel dog',
        'affenpinscher dog',
        'lakeland terrier dog',
        'clumber spaniel dog',
        'bedlington terrier dog',
        'australian terrier dog',
        'tibetan mastiff dog',
        'norfolk terrier dog',
        'tibetan spaniel dog',
        'russian black terrier dog',
        'german spitz dog',
    ],
    "Scenes": [
        'abbey',
        'waterfall_cascade',
        'velodrome_outdoor',
        'volleyball_court_outdoor',
        'arena_hockey',
        'arena_basketball',
        'terrace_farm',
        'tree_farm',
        'tundra',
        'valley',
        'stone_circle',
        'volcano',
        'waterfall_cataract',
        'pavilion',
        'waterfall_fan',
        'waterfall_plunge',
        'watering_hole',
        'wave',
        'wheat_field',
        'waterfall_block',
        'bamboo_forest',
        'snowfield',
        'sea_cliff',
        'moor',
        'velodrome_indoor',
        'track_outdoor',
        'track_indoor',
        'tennis_court_outdoor',
        'athletic_field_outdoor',
        'badminton_court_indoor',
        'badminton_court_outdoor',
        'baseball_field',
        'basketball_court_indoor',
        'basketball_court_outdoor',
        'batters_box',
        'batting_cage_indoor',
        'batting_cage_outdoor',
        'boxing_ring',
        'bullpen',
        'football_field',
        'ice_skating_rink_indoor',
        'martial_arts_gym',
        'pitchers_mound',
        'soccer_field',
        'squash_court',
        'stadium_baseball',
        'stadium_football',
        'stadium_soccer',
        'tennis_court_indoor',
        'mountain',
        'mountain_path',
        'mountain_snowy',
        'observatory_indoor',
        'nursing_home',
        'packaging_plant',
        'pagoda',
        'palace',
        'pantry',
        'pier',
        'picnic_area',
        'piano_store',
        'physics_laboratory',
        'phone_booth',
        'pharmacy',
        'pet_shop',
        'jail_indoor',
        'pedestrian_overpass_outdoor',
        'patio',
        'particle_accelerator',
        'parlor',
        'parking_lot',
        'parking_garage_outdoor',
        'parking_garage_indoor',
        'parade_ground',
        'oast_house',
        'observatory_outdoor',
        'oasis',
        'office',
        'ocean',
        'orchard',
        'ski_slope',
        'outcropping',
        'pasture',
        'pond',
        'rainforest',
        'river',
        'rock_arch',
        'sandbar',
        'savanna',
        'park',
        'badlands',
        'ossuary',
        'organ_loft_exterior',
        'optician',
        'operating_room',
        'oilrig',
        'oil_refinery_outdoor',
        'office_cubicles',
        'office_building',
        'wrestling_ring_indoor',
    ],
    "People": [
        'George_W_Bush',
        'Colin_Powell',
        'Tony_Blair',
        'Donald_Rumsfeld',
        'Ariel_Sharon',
        'Junichiro_Koizumi',
        'John_Ashcroft',
        'Jacques_Chirac',
        'Serena_Williams',
        'Vladimir_Putin',
        'Gloria_Macapagal_Arroyo',
        'Arnold_Schwarzenegger',
        'Jennifer_Capriati',
        'Lleyton_Hewitt',
        'Laura_Bush',
        'Alejandro_Toledo',
        'Andre_Agassi',
        'Silvio_Berlusconi',
        'Tom_Ridge',
        'Megawati_Sukarnoputri',
        'Vicente_Fox',
        'Roh_Moo-hyun',
        'David_Beckham',
        'John_Negroponte',
        'Guillermo_Coria',
        'Mahmoud_Abbas',
        'Jack_Straw',
        'Juan_Carlos_Ferrero',
        'Ricardo_Lagos',
        'Gray_Davis',
        'Tom_Daschle',
        'Atal_Bihari_Vajpayee',
        'Winona_Ryder',
        'Tiger_Woods',
        'Lindsay_Davenport',
        'Naomi_Watts',
        'Pete_Sampras',
        'Jennifer_Lopez',
        'Jennifer_Aniston',
        'Carlos_Menem',
        'Angelina_Jolie',
        'Igor_Ivanov',
        'Julianne_Moore',
        'John_Howard',
        'Joschka_Fischer',
        'Nicole_Kidman',
        'Tim_Henman',
        'Lance_Armstrong',
        'Michael_Schumacher',
        'Jean_Charest',
        'Spencer_Abraham',
        'Venus_Williams',
        'Trent_Lott',
        'Halle_Berry',
        'Dominique_de_Villepin',
        'Meryl_Streep',
        'Pierce_Brosnan',
        'Andy_Roddick',
        'Norah_Jones',
        'Kim_Clijsters',
        'David_Nalbandian',
        'Roger_Federer',
        'James_Blake',
        'Britney_Spears',
        'Edmund_Stoiber',
        'Salma_Hayek',
        'Jackie_Chan',
        'Joe_Lieberman',
        'Jennifer_Garner',
        'Michael_Jackson',
        'Jeb_Bush',
        'Harrison_Ford',
        'Adrien_Brody',
        'Howard_Dean',
        'Rubens_Barrichello',
        'Anna_Kournikova',
        'Mike_Weir',
        'Mark_Philippoussis',
        'Ian_Thorpe',
        'Muhammad_Ali',
        'Kate_Hudson',
        'Colin_Farrell',
        'Ray_Romano',
        'Maria_Shriver',
        'Justin_Timberlake',
        'Bob_Hope',
        'Robert_Blake',
        'Amelia_Vega',
        'Clay_Aiken',
        'Zinedine_Zidane',
        'Valentino_Rossi',
        'Boris_Becker',
        'Elsa_Zylberstein',
        'Lance_Bass',
        'Natalie_Maines',
        'Ludivine_Sagnier',
        'George_Lopez',
        'Martina_McBride',
        'Michael_Chiklis',
        'Steffi_Graf'
    ],
}
domain_model = ["Stable Diffusion 1.4"]
# domain_mp, domain_me, domain_s, domain_l (pretty names) are all derived further down from the
# declarative registries (MP_REGISTRY/S_REGISTRY/L_REGISTRY) and from `type_me` itself, once the
# corresponding `type_*` Literals below exist.

# Types (as they appear in the code/files)
type_unlearning_algorithm = Literal["distil", "munba", "uce"]
type_task = Literal["breeds", "scenes", "people"]
type_model = Literal["sd1.4"]
type_mp = Literal["brisque_diff", "clip_diff", "rmse", "ssim", "dino_diff"]
type_me = Literal[
    "Emitter worst interfered brisque diff",
    "Emitter worst interfered clip diff",
    "Emitter worst interfered rmse",
    "Emitter worst interfered ssim",
    "Emitter number of interfered worse than target brisque diff",
    "Emitter number of interfered worse than target clip diff",
    "Emitter number of interfered worse than target rmse",
    "Emitter number of interfered worse than target ssim",
    "Emitter number of interfered worse than zero clip diff",
    "Emitter average brisque diff",
    "Emitter average clip diff",
    "Emitter average rmse",
    "Emitter average ssim",
    "Receiver worst interfered brisque diff",
    "Receiver worst interfered clip diff",
    "Receiver worst interfered rmse",
    "Receiver worst interfered ssim",
    "Receiver number of interfered worse than target brisque diff",
    "Receiver number of interfered worse than target clip diff",
    "Receiver number of interfered worse than target rmse",
    "Receiver number of interfered worse than target ssim",
    "Receiver number of interfered worse than zero clip diff",
    "Receiver average brisque diff",
    "Receiver average clip diff",
    "Receiver average rmse",
    "Receiver average ssim",
    "Emitter minus receiver worst interfered brisque diff",
    "Emitter minus receiver worst interfered clip diff",
    "Emitter minus receiver worst interfered rmse",
    "Emitter minus receiver worst interfered ssim",
    "Emitter minus receiver number of interfered worse than target brisque diff",
    "Emitter minus receiver number of interfered worse than target clip diff",
    "Emitter minus receiver number of interfered worse than target rmse",
    "Emitter minus receiver number of interfered worse than target ssim",
    "Emitter minus receiver number of interfered worse than zero clip diff",
    "Emitter minus receiver average brisque diff",
    "Emitter minus receiver average clip diff",
    "Emitter minus receiver average rmse",
    "Emitter minus receiver average ssim",
    "Emitter worst interfered dino diff",
    "Emitter number of interfered worse than target dino diff",
    "Emitter average dino diff",
    "Receiver worst interfered dino diff",
    "Receiver number of interfered worse than target dino diff",
    "Receiver average dino diff",
    "Emitter minus receiver worst interfered dino diff",
    "Emitter minus receiver number of interfered worse than target dino diff",
    "Emitter minus receiver average dino diff",
    "Embedding specificity ratio",
    "Forget clip diff",
    "Retain average clip diff",
]
type_s = Literal[
    "clip",
    "jacc",
    "dino",
    "act",
    "weight_overlap",  # cosine similarity of trained LoRA weight changes (B@A); scenes/distil only
]

type_l = Literal[
    "clip_embedding",
    "dino_embedding",
]

type_regression_algorithm = Literal[
    "linear_regression",
    "random_forest",
]


def model_segment(model: type_model) -> str:
    """Filename/folder segment for the base model.

    ``sd1.4`` (the historical default and only produced model) keeps every existing asset
    name unchanged by returning an empty segment; any other base model adds a disambiguating
    ``_{model}`` segment so its assets never collide with the ``sd1.4`` ones. This mirrors
    ``metadata._embedding_function_suffix`` for the embedding-function dimension. It is
    interface-only for now: no non-``sd1.4`` model is produced, but the naming is ready for
    one without renaming any current file.

    Note this does NOT apply to the activation-fingerprint files, which already embed the
    model name explicitly (``act_fingerprints_{task}_{model}.json``, non-empty for ``sd1.4``);
    those keep their own convention untouched.
    """
    return "" if model == "sd1.4" else f"_{model}"


# =============================================================================
# DIRECTION CONVENTION  (read carefully — these dicts were historically MIS-commented)
# -----------------------------------------------------------------------------
# The arrow points toward the metric's HEALTHY / LESS-interference direction
# (i.e. the direction of *better* image quality / *more* preserved knowledge).
# The arrow does NOT point toward "more interference".
#
#     "↑"  ->  HIGHER value = LESS interference (better).  MORE interference = LOWER value.
#     "↓"  ->  LOWER  value = LESS interference (better).  MORE interference = HIGHER value.
#
# This is exactly how the code consumes it everywhere (result_templates.py:768, 2839):
#     is_worst_biggest = mp_to_direction[mp] != '↑'
#     # worst == most interference.  For '↑' metrics worst = SMALLEST value;
#     #                              for '↓' metrics worst = BIGGEST value.
# =============================================================================
type_direction = Literal["↑", "↓"]


# =============================================================================
# Declarative registries — single source of truth per concept.
# -----------------------------------------------------------------------------
# Each record below replaces what used to be 2-4 independently hand-maintained lists/dicts
# (a pretty-name list, a direction dict, a GUI_TO_BACKEND entry, ...). Adding a metric, an
# embedding function, or an unlearning method is exactly one new entry in the corresponding
# registry; `domain_*`, `*_to_direction`, and `GUI_TO_BACKEND` below are all *derived* from it,
# so there is nothing else to keep in sync. The `type_*` Literals stay hand-written (mypy needs
# a static definition), but every registry key is one of that Literal's members, and a
# completeness test (`tests/test_configuration.py`) locks the two together.
# =============================================================================

_K = TypeVar("_K")


def _pretty_names(registry: Dict[_K, "MetricWithDirectionSpec"], order: List[_K]) -> List[str]:
    """Look up ``order`` in ``registry`` and return the (guaranteed non-``None``) pretty names,
    in exactly the given order. Raises if a key is missing a display name (not GUI-exposed)."""
    names: List[str] = []
    for key in order:
        pretty = registry[key].name_pretty
        assert pretty is not None, f"{key!r} has no display name (not GUI-exposed)"
        names.append(pretty)
    return names


class MetricWithDirectionSpec(BaseModel):
    """One (software name, display name, direction) record for a metric used either as a
    per-pair interference dimension (`type_mp`) or a similarity dimension (`type_s`).

    ``name_pretty`` is ``None`` for a metric that is computed and typed but intentionally not
    GUI-selectable (currently only `weight_overlap`, a LoRA-weight-specific diagnostic).
    """
    name: str
    name_pretty: Optional[str] = None
    direction: type_direction


class LSpec(BaseModel):
    """One (software name, display name) record for an embedding function (`type_l`)."""
    name: str
    name_pretty: str


class UnlearningAlgorithmSpec(BaseModel):
    """One (software name, display name) record for an unlearning method
    (`type_unlearning_algorithm`)."""
    name: str
    name_pretty: str


MP_REGISTRY: Dict[type_mp, MetricWithDirectionSpec] = {
    # clip_diff = clip_on - clip_off.  off = original/baseline model, on = unlearned model.
    # Damaging a receiver lowers clip_on => clip_diff goes NEGATIVE.
    # ==> MORE interference = MORE NEGATIVE (lower) clip_diff.  Arrow "↑" = healthy/higher.
    "clip_diff": MetricWithDirectionSpec(name="clip_diff", name_pretty="Delta Clip", direction="↑"),
    # brisque_diff = brisque_on - brisque_off.  BRISQUE: lower = better quality, so damage
    # raises brisque_on => positive diff.  ==> MORE interference = HIGHER (more positive).
    "brisque_diff": MetricWithDirectionSpec(name="brisque_diff", name_pretty="Delta Brisque", direction="↓"),
    # rmse between off/on images.  Identical images => 0.  ==> MORE interference = HIGHER rmse.
    "rmse": MetricWithDirectionSpec(name="rmse", name_pretty="RMSE", direction="↓"),
    # ssim between off/on images, in [0,1].  Identical => 1.  ==> MORE interference = LOWER ssim.
    "ssim": MetricWithDirectionSpec(name="ssim", name_pretty="SSIM", direction="↑"),
    # dino_diff = DINOv2 cosine similarity of off/on images, in [0,1].  Identical => 1.
    # ==> MORE interference = LOWER dino_diff (same polarity as ssim).
    "dino_diff": MetricWithDirectionSpec(name="dino_diff", name_pretty="DINO Cosine Similarity", direction="↑"),
}
# GUI dropdown order (Forgety's frontend) — intentionally NOT the same order as MP_REGISTRY /
# type_mp above; preserved exactly as before this refactor.
_MP_DISPLAY_ORDER: List[type_mp] = ["clip_diff", "brisque_diff", "rmse", "ssim", "dino_diff"]

# Similarity metrics: arrow "↑" = HIGHER value means MORE similar (all current s metrics
# are similarities, so all are "↑").  Higher similarity is hypothesised to predict more
# interference, but that is a hypothesis about the s<->m_p relationship, NOT the polarity
# of s itself.
S_REGISTRY: Dict[type_s, MetricWithDirectionSpec] = {
    "clip": MetricWithDirectionSpec(name="clip", name_pretty="Clip Cosine Similarity", direction="↑"),
    "jacc": MetricWithDirectionSpec(name="jacc", name_pretty="Jacc Similarity", direction="↑"),
    "dino": MetricWithDirectionSpec(name="dino", name_pretty="DINOv2 Cosine Similarity", direction="↑"),
    "act": MetricWithDirectionSpec(name="act", name_pretty="UNet Cross-Attention Similarity", direction="↑"),
    # No name_pretty: cosine similarity of trained LoRA weight changes (B@A), scenes/distil
    # only — a diagnostic, not GUI-selectable (not in GUI_TO_BACKEND['similarity_metric']).
    "weight_overlap": MetricWithDirectionSpec(name="weight_overlap", direction="↑"),
}
_S_DISPLAY_ORDER: List[type_s] = ["clip", "jacc", "dino", "act"]  # weight_overlap excluded: no display name

L_REGISTRY: Dict[type_l, LSpec] = {
    "clip_embedding": LSpec(name="clip_embedding", name_pretty="Clip Embedding"),
    "dino_embedding": LSpec(name="dino_embedding", name_pretty="DINOv2 Embedding"),
}
_L_DISPLAY_ORDER: List[type_l] = ["clip_embedding", "dino_embedding"]

ALGORITHM_REGISTRY: Dict[type_unlearning_algorithm, UnlearningAlgorithmSpec] = {
    "distil": UnlearningAlgorithmSpec(name="distil", name_pretty="SPARE"),
    "munba": UnlearningAlgorithmSpec(name="munba", name_pretty="Munba"),
    "uce": UnlearningAlgorithmSpec(name="uce", name_pretty="UCE"),
}
_UNLEARNING_ALGORITHM_DISPLAY_ORDER: List[type_unlearning_algorithm] = ["distil", "munba", "uce"]

domain_unlearning_algorithm = [ALGORITHM_REGISTRY[k].name_pretty for k in _UNLEARNING_ALGORITHM_DISPLAY_ORDER]
domain_mp = _pretty_names(MP_REGISTRY, _MP_DISPLAY_ORDER)
domain_s = _pretty_names(S_REGISTRY, _S_DISPLAY_ORDER)
domain_l = [L_REGISTRY[k].name_pretty for k in _L_DISPLAY_ORDER]

mp_to_direction: Dict[type_mp, type_direction] = {k: v.direction for k, v in MP_REGISTRY.items()}
s_to_direction: Dict[type_s, type_direction] = {k: v.direction for k, v in S_REGISTRY.items()}
# me_to_direction can... be infered?

# And converting between them
GUI_TO_BACKEND: Dict[str, Dict[str, str]] = {
    "unlearning_algorithm": {v.name_pretty: k for k, v in ALGORITHM_REGISTRY.items()},
    "task": {
        "Breeds": "breeds",
        "Scenes": "scenes",
        "People": "people",
    },
    "model": {
        "Stable Diffusion 1.4": "sd1.4",
    },
    "interference_pair": {v.name_pretty: k for k, v in MP_REGISTRY.items() if v.name_pretty is not None},
    "similarity_metric": {v.name_pretty: k for k, v in S_REGISTRY.items() if v.name_pretty is not None},
    "latent_embedding": {v.name_pretty: k for k, v in L_REGISTRY.items()},
}


# =============================================================================
# type_me — combinatorial structure.
# -----------------------------------------------------------------------------
# The 51 `type_me` names are not arbitrary: {Emitter, Receiver, Emitter minus receiver} x
# {worst interfered, number of interfered worse than target, average} x each base Mp metric,
# plus a clip_diff-only "worse than zero" variant per role, the dino_diff group (retrofitted
# later — no "worse than zero" variant), and three named specials. `generate_me_names()`
# reproduces this set from the same MP_REGISTRY above; a test locks it against `type_me`'s own
# members. This also gives `choose_metric_column_interference_per_entity` (metadata.py) a single
# authoritative naming contract instead of a bare regex reverse-engineering a convention defined
# nowhere. The irregularities baked in below are historical, not principled, and are encoded
# explicitly rather than idealised away — changing them would add/remove `type_me` members,
# which is out of scope of this refactor (see CONTRIBUTING.md's backward-compatibility rule).
# =============================================================================
_ME_ROLES = ["Emitter", "Receiver", "Emitter minus receiver"]
_ME_AGGREGATIONS = ["worst interfered", "number of interfered worse than target", "average"]
_ME_BASE_MP_ORDER: List[type_mp] = ["brisque_diff", "clip_diff", "rmse", "ssim"]  # dino_diff handled separately below


def generate_me_names() -> List[str]:
    """Reproduce the `type_me` vocabulary from its combinatorial structure. See the module-level
    comment above this function for the two historical irregularities encoded here."""
    names: List[str] = []
    for role in _ME_ROLES:
        for aggregation in _ME_AGGREGATIONS:
            for mp in _ME_BASE_MP_ORDER:
                names.append(f"{role} {aggregation} {mp.replace('_', ' ')}")
        # Only clip_diff has a "worse than zero" variant (one per role).
        names.append(f"{role} number of interfered worse than zero clip diff")
    # dino_diff was retrofitted after the base four metrics: no "worse than zero" variant.
    for role in _ME_ROLES:
        for aggregation in _ME_AGGREGATIONS:
            names.append(f"{role} {aggregation} dino diff")
    names += ["Embedding specificity ratio", "Forget clip diff", "Retain average clip diff"]
    return names


# domain_me is the same 51 pretty names as `type_me` (they ARE the pretty names — `type_me`
# values are already display-cased). Read directly off the Literal's own members instead of
# hand-duplicating the list a second time, so the two can never drift.
domain_me = list(get_args(type_me))


task_to_attributes_of_interest = {
    "breeds": [
        "grooming_frequency_category_binary",  # (a continuous attribute describing how often brushing is required; discretized into two bins based on quartiles)
        "group",  # (categorical, with values such as retrievers and terriers)
    ],
    "scenes": [
        "sports",
        "natural",
    ],
    "people": [
        "hpi_bin",
        "occupation_simplified",
    ]
}


# The method's hyperparameter is specially problematic to map/find, because i initially thought it should
# be configurable and then I chagned my mind. Sometimes it can be infered (like in `choose_metric_column_interference_per_entity`),
# Also, in the actual slurm scripts they are hardcoded all over
# But sometimes I use this hardcoded mapping:
unlearning_algorithm_to_epochs = {
    'breeds': {
        'distil': 100,
        'munba': 50,
        'uce': 0,
    },
    'scenes': {
        'distil': 100,
        'munba': 100,
        'uce': 0,
    },
    'people': {
        'distil': 400,
        'munba': 200,
        'uce': 0,
    },
}



def convert_params_from_gui_to_backend(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert GUI values to backend literal values.
    Unknown keys are passed through unchanged.
    None stays None.
    """
    converted_params: Dict[str, Any] = {}

    for key, value in params.items():
        if value is None:
            converted_params[key] = None
            continue

        mapping = GUI_TO_BACKEND.get(key)
        if mapping is not None:
            converted_params[key] = mapping.get(value, value)
        else:
            converted_params[key] = value

    return converted_params



