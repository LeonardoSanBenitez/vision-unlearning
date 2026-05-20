# Heavy submodules (datasets, evaluator, metrics) require torch and torchvision.
# Guard the star-imports so that torch-free environments (e.g. the forgety web
# container, which only uses vision_unlearning.benchmarks.I_care) can import the
# package without crashing.  In environments where torch IS installed, behaviour
# is identical to before.
try:
    from vision_unlearning.datasets import *
    from vision_unlearning.evaluator import *
    from vision_unlearning.metrics import *
except ImportError:
    pass

# integrations and utils have empty __init__.py files (no star-exports), so
# these lines are no-ops but we keep them for completeness.
from vision_unlearning.integrations import *
from vision_unlearning.utils import *
