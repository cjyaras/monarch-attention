import json
import os
import uuid
from typing import Dict

from transformers.configuration_utils import PretrainedConfig


def generate_random_file_name(extension="json"):
    random_string = uuid.uuid4().hex
    return f"{random_string}.{extension}"


class Logger:
    def __init__(self, save_folder: str):
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def save(self, config: PretrainedConfig, result: Dict[str, float]) -> str:
        file_name = generate_random_file_name()
        save_path = os.path.join(self.save_folder, file_name)
        with open(save_path, "w") as f:
            json.dump({"attention_type": config.attention_type, "result": result}, f)

        return file_name

    def load(self, file_name: str) -> Dict:
        save_path = os.path.join(self.save_folder, file_name)
        with open(save_path, "r") as f:
            return json.load(f)
