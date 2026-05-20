# Guard heavy star-imports so that torch-free environments (e.g. the forgety
# web container) can import submodules such as datasets.testbed without
# triggering torch/torchvision at package-init time.  When torch IS installed
# the behaviour is identical to before.
try:
    from vision_unlearning.datasets.base import *
    from vision_unlearning.datasets.cifar import *
    from vision_unlearning.datasets.coco import *
    from vision_unlearning.datasets.imagenette import *
    from vision_unlearning.datasets.local import *
    from vision_unlearning.datasets.others import *
except ModuleNotFoundError:
    pass

# testbed is intentionally NOT guarded: it is needed by I_care.metadata and
# does not import torch at module level.
from vision_unlearning.datasets.testbed import *
