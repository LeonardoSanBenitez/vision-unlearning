from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, ConfigDict
from huggingface_hub.repocard_data import EvalResult
from vision_unlearning.utils.logger import get_logger


logger = get_logger('unlearner')


class Unlearner(BaseModel, ABC):
    '''
    performs the actual finetuning

    One unlearner may have variations/parametrizations that correspond to different unlearning algorithms/methods
    '''

    # An unrecognised keyword argument is an error, not something to drop silently. Pydantic's
    # default is to ignore extras, and two hyperparameters rode on that for the whole life of this
    # class: `lora_dropout`, passed as 0.2 by every caller and never reaching the adapter, and
    # `train_batch_size`, whose real name is `per_device_train_batch_size` and which happened to be
    # set to the same value everywhere, so nothing ever looked wrong.
    model_config = ConfigDict(extra="forbid")

    # Whether the evaluation step displays its comparison figures interactively.
    #
    # Off by default because an unlearner is normally run unattended, and showing a figure calls
    # `plt.show()`, which blocks until a window is closed whenever matplotlib has an interactive
    # backend. A training run from a script or a scheduler then hangs forever with no error and no
    # output -- and the hang is invisible under a headless backend, so it survives continuous
    # integration and appears only on a desktop.
    #
    # Set it True from a notebook, where blocking is exactly what you want. The figures are returned
    # from `evaluate()` either way, so leaving it off costs nothing but the window.
    plot_show: bool = False

    @abstractmethod
    def train(self) -> List[EvalResult]:
        pass
