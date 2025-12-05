from abc import ABC
from typing import Optional
from pydantic import BaseModel, ConfigDict
import torch


class Metric(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    device: Optional[torch.device|str] = None
