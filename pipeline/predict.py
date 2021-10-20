from modeling.classifier import MultiHeadsClassifier
from transformers import AutoTokenizer
import torch
import os
import numpy as np
from huggingface_hub import snapshot_download
from utils.util_functions import read_json
from typing import Tuple


class ClassificationPipeline:
    def __init__(self, model, tokenizer, label_map, max_len, use_gpu=True):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        self.device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    @staticmethod
    def _read_checkpoint_folder(checkpoint_path: str) -> Tuple[str, dict, dict]:
        model_path = os.path.join(checkpoint_path, "multiHeadsClassifier.bin")
        model_configs = read_json(os.path.join(checkpoint_path, "network_configs.json"))
        label_map = read_json(os.path.join(checkpoint_path, "label_map.json"))
        return model_path, model_configs, label_map

    @classmethod
    def from_hub(cls, repo_id: str, use_gpu: bool = False):
        checkpoint_path = snapshot_download(repo_id)
        device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        model_path, model_configs, label_map = cls._read_checkpoint_folder(checkpoint_path)

        # TODO: make the save/from_local for tokenizer better
        tokenizer = AutoTokenizer.from_pretrained(model_configs["tokenizer_name"])

        model = MultiHeadsClassifier.load(model_path, model_configs, device)
        return cls(model, tokenizer, label_map, model_configs["max_len"], use_gpu)

    @classmethod
    def from_local(cls, checkpoint_path: str, use_gpu: bool = True):
        model_path, model_configs, label_map = cls._read_checkpoint_folder(checkpoint_path)

        tokenizer = AutoTokenizer.from_pretrained(model_configs["tokenizer_name"])
        device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        model = MultiHeadsClassifier.load(model_path, model_configs, device)
        return cls(model, tokenizer, label_map, model_configs["max_len"], use_gpu)

    def _prepare_inputs(self, message: str):
        return self.tokenizer.encode_plus(
            message,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )

    def predict(self, text: str):
        self.model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(text).to(self.device)
            probabilities = [prob.tolist() for prob in self.model.predict(inputs)]
        preds = [(np.max(prob), np.argmax(prob)) for prob in probabilities]
        predicted_classes = {
            "text": text,
            **{f"Node-{index + 1}": {
                self.label_map[f"level{index + 1}"][str(pred[1])]: round(float(pred[0]), 2)
            } for index, pred in enumerate(preds)}}
        return predicted_classes


if __name__ == '__main__':
    predictor = ClassificationPipeline.from_hub(repo_id="GUOKP/sentiment_model", use_gpu=False)
    result = predictor.predict(text="The price of the pencil is 5 dollars.")

    import pprint
    pprint.pprint(result)
