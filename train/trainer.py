import os
import torch
import torch.nn as nn
from tqdm import tqdm

from evaluate.evaluator import eval_epoch
from utils.util_functions import write_json
from utils.console import console


def train_epoch(model, dataloader, loss_fn, optimizer, device, scheduler, num_heads=4):
    model = model.train()
    losses = []
    correct_predictions = [0] * num_heads

    for d in tqdm(dataloader, desc="Batches in one epoch: "):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = [d[f"label{idx + 1}"].to(device) for idx in range(num_heads)]
        outputs = model({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

        # level 1 => full labels
        # level 2 to 4 => lots of empty labels...
        labels_fulfilled = [None, *[labels[idx][labels[idx] != -1] for idx in range(1, num_heads)]]
        loss_batch = 0
        for idx in range(num_heads):
            _, preds_label = torch.max(outputs[idx], dim=1)
            if idx == 0:
                loss_batch = loss_fn(outputs[idx], labels[idx])
            elif labels_fulfilled[idx].shape[0] != 0:
                loss_batch += loss_fn(outputs[idx][labels[idx] != -1], labels_fulfilled[idx])
            correct_predictions[idx] += torch.sum(preds_label == labels[idx]).item()

        losses.append(loss_batch.item())

        # update the parameters
        loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


class GTrainer:
    def __init__(self, model, tokenizer, device, optimizer, scheduler, loss_fn, max_len, num_heads, num_labels):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_labels = num_labels
        # record the accuracy during training.
        self.history = {
            "train_loss": [],
            "validation_loss": [],
            **{f"acc_level_{idx + 1}": {"train": {}, "validation": {}} for idx in range(num_heads)},
        }

    def train(self, num_epochs, train_dataloader, df_train,
              eval_dataloader=None, df_eval=None, do_eval=False):
        """
        Train the neural network
        :param num_epochs: number of epochs to train the network
        :param train_dataloader: train dataloader
        :param df_train: train dataframe
        :param eval_dataloader: evaluation dataloader, need set do_eval to True
        :param df_eval: evaluation dataframe, need set do_eval to True
        :param do_eval: flag => perform evaluation or not
        :return:
        """
        tasks = [epoch for epoch in range(1, num_epochs + 1)]
        with console.status("Training the Neural Network") as _:
            while tasks:
                epoch = tasks.pop(0)
                train_epoch(self.model, train_dataloader, self.loss_fn, self.optimizer, self.device, self.scheduler,
                            num_heads=self.num_heads)
                train_acc, train_loss, train_n_examples = eval_epoch(self.model, train_dataloader, self.loss_fn,
                                                                     self.device, df_train,
                                                                     num_heads=self.num_heads)
                self.history["train_loss"].append(train_loss)
                for level in range(1, self.num_heads + 1):
                    self.history[f"acc_level_{level}"]["train"][f"epoch-{epoch}"] = train_acc[level]
                console.log(f"Finished Epoch-{epoch}, train accuracy: {train_acc}, train loss: {train_loss}")
                if do_eval:
                    assert eval_dataloader is not None
                    assert df_eval is not None
                    self.evaluate(eval_dataloader, df_eval)

    def evaluate(self, eval_dataloader, df_eval):
        eval_acc, eval_loss, eval_n_examples = eval_epoch(self.model, eval_dataloader, self.loss_fn, self.device,
                                                          df_eval, num_heads=self.num_heads)
        console.log(f"Epoch Evaluation, accuracy: {eval_acc}, loss: {eval_loss}")

    def save_model_tokenizer(self, output_dir, label_map):
        os.makedirs(output_dir, exist_ok=True)
        configs = {
            "max_len": int(self.max_len),
            "num_heads": int(self.num_heads),
            "num_labels": [int(label) for label in self.num_labels],
            "tokenizer_name": str(self.tokenizer.name_or_path),
            **self.history
        }
        write_json(configs, os.path.join(output_dir, "network_configs.json"))
        write_json(label_map, os.path.join(output_dir, "label_map.json"))
        torch.save(self.model.state_dict(), os.path.join(output_dir, "multiHeadsClassifier.bin"))
