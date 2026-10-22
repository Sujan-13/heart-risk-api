from typing import Literal
from pydantic import BaseModel, Field

class PatientIn(BaseModel):
    Age: int = Field(ge=18, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "ASY", "TA"]
    RestingBP: int = Field(ge=0, le=300)
    Cholesterol: int = Field(ge=0, le=700)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=40, le=250)
    ExerciseAngina: Literal["N", "Y"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]
    