from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import yaml

class RunMetadata(BaseModel):
    test_size: float
    output_dir: Path
    model_type: str
    features_used: List[str]
    timestamp: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True  # allow Path types

def write_metadata_yaml(metadata: RunMetadata, output_path: Path) -> None:
    output_path = output_path / "run_metadata.yaml"

    metadata_dict = metadata.dict()

    with output_path.open("w") as f:
        yaml.safe_dump(metadata_dict, f, sort_keys=False)