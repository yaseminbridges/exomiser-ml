from dataclasses import dataclass, field

from pathlib import Path
from typing import List, Union
from datetime import datetime
import yaml
from serde import to_dict, serde


@serde
@dataclass
class RunMetadata:
    test_size: Union[float, None]
    output_dir: str
    model_type: str
    training_data: str
    test_dir: str
    features_used: List[str]
    timestamp: datetime = field(default_factory=datetime.now)



def write_metadata_yaml(metadata: RunMetadata, output_path: Path) -> None:
    output_path = output_path / "run_metadata.yaml"
    with open(output_path, "w") as metadata_file:
        yaml.dump(to_dict(metadata), metadata_file, sort_keys=False, default_style="")
    metadata_file.close()