from pydantic import BaseModel, ConfigDict
from abc import ABC


class Metric(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
