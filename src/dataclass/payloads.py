from pydantic import BaseModel
from typing import Optional, List
from .controlnet import ControlNetScript

class InputPayload(BaseModel):
    preset: str
    user_input: str
    n_output: Optional[int] = 3

class SdFwdPayload(BaseModel):
    prompt: str
    negative_prompt: str
    batch_size: Optional[int] = 1
    sample_index: Optional[str] = "Euler a"
    seed: Optional[int] = -1
    steps: Optional[int] = 20
    width: Optional[int] = 512
    height: Optional[int] = 512
    cfg_scale: Optional[int] = 7

class SdFwdPaylodWithImage(SdFwdPayload):
    alwayson_scripts: ControlNetScript

class OutputPayload(BaseModel):
    imgs: List[str]