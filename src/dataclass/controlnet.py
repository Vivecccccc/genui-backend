from pydantic import BaseModel
from typing import Optional, List

class ControlNetArgs(BaseModel):
    input_image: str
    module: Optional[str] = "canny"
    model: Optional[str] = "control_v11p_sd15_canny [d14c016b]"
    resize_mode: Optional[str] = "Crop and Resize"
    processor_res: Optional[int] = 512
    threshold_a: Optional[int] = 100
    threshold_b: Optional[int] = 200
    weight: Optional[float] = 1.0

class ControlNet(BaseModel):
    args: List[ControlNetArgs]

class ControlNetScript(BaseModel):
    controlnet: ControlNet