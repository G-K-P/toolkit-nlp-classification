import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoConfig
from scipy.special import softmax
from huggingface_hub import snapshot_download
from utils.util_functions import read_json


class MultiHeadsClassifier(nn.Module):

    def __init__(self, model_checkpoint="distilbert-base-uncased", num_heads=4, num_labels=None,
                 output_hidden_states=False, dropout=0.1):
        super(MultiHeadsClassifier, self).__init__()
        if num_labels is None:
            raise ValueError("num_labels can't be empty.")
        self.num_heads = num_heads
        self.lm_config = AutoConfig.from_pretrained(model_checkpoint, output_hidden_states=output_hidden_states)
        self.language_model = AutoModel.from_pretrained(model_checkpoint, config=self.lm_config)
        # Freeze the LM
        # for param in self.language_model.parameters():
        #     param.requires_grad = False
        self.pre_classifier = nn.Linear(768, 768, bias=True)
        self.sigmoid = nn.Sigmoid()
        assert num_heads == len(num_labels)
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            if i == 0:
                self.heads.append(nn.Linear(768, num_labels[i]))
            else:
                self.heads.append(nn.Linear(768 + num_labels[i - 1], num_labels[i]))
        self.dropout = nn.Dropout(p=dropout)

    @classmethod
    def from_hub(cls, repo_id, use_gpu=False):
        checkpoint_path = snapshot_download(repo_id)
        model_path = os.path.join(checkpoint_path, "multiHeadsClassifier.bin")
        model_configs = read_json(os.path.join(checkpoint_path, "network_configs.json"))

        # TODO: make the save/from_local for tokenizer better
        device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        model = cls(model_configs["tokenizer_name"],
                    num_heads=model_configs["num_heads"],
                    num_labels=model_configs["num_labels"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    @classmethod
    def load(cls, model_path, model_configs, device):
        """
            model_path: where the model is stored
            model_configs: the configs saved in the network_configs.json file in model folder
        """
        model = cls(model_configs["tokenizer_name"],
                    num_heads=model_configs["num_heads"],
                    num_labels=model_configs["num_labels"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def predict(self, inputs):
        levels_output = [output.detach().cpu() for output in self.forward(inputs)]
        levels_softmax = [softmax(level_output) for level_output in levels_output]
        return levels_softmax

    def forward(self, inputs):
        # pass through the LM, get the last transformer layer's output
        outputs = self.language_model(**inputs)

        # average over the representations of tokens
        # outputs = torch.mean(outputs[0], dim=1)

        # use the [CLS] token to classify
        outputs = outputs["last_hidden_state"][:, 0]

        outputs = self.pre_classifier(outputs)
        outputs = self.sigmoid(outputs)
        levels_output = []
        for head in self.heads:
            if not levels_output:
                levels_output.append(self.dropout(head(outputs)))
            else:
                levels_output.append(self.dropout(head(torch.cat((outputs, levels_output[-1]), dim=1))))
        return levels_output
