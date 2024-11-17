import importlib
import os

import yaml

from bmfm_sm.core.data_modules.namespace import Modality


class ModelRegistry:
    @staticmethod
    def get_checkpoint(modality):
        if type(modality) is str and modality.upper() in Modality.__members__:
            modality = Modality[modality.upper()]

        with importlib.resources.open_text(
            "bmfm_sm.resources", "pretrained_ckpts.yaml"
        ) as f:
            config = yaml.safe_load(f)
            relative_path = config.get(modality.name)

            if relative_path:
                base_path = os.environ.get("BMFM_HOME")
                if base_path is None:
                    raise ValueError(
                        "Remember to set the BMFM_HOME environment variable"
                    )

                checkpoint_path = os.path.join(
                    base_path, "bmfm_model_dir/pretrained/", relative_path
                )
                return checkpoint_path

            raise ValueError(f"Checkpoint not available for modality '{modality}'")
