from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    prediction:       str          = Field(..., description="NORMAL or PNEUMONIA")
    confidence:       float        = Field(..., description="Probability of predicted class (0-1)")
    prob_normal:      float        = Field(..., description="Probability of NORMAL class")
    prob_pneumonia:   float        = Field(..., description="Probability of PNEUMONIA class")
    gradcam_image:    str          = Field(..., description="Base64-encoded GradCAM overlay PNG")
    low_confidence:   bool         = Field(..., description="True if confidence < threshold warning")
    message:          Optional[str] = Field(None, description="Clinical warning if low confidence")

    model_config = {"json_schema_extra": {
        "example": {
            "prediction":     "PNEUMONIA",
            "confidence":     0.9823,
            "prob_normal":    0.0177,
            "prob_pneumonia": 0.9823,
            "gradcam_image":  "<base64 string>",
            "low_confidence": False,
            "message":        None,
        }
    }}


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    device:       str
    uptime_sec:   float
    model_path:   str
