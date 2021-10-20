# NLP toolkit
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
# my libraries
from modeling.classifier import MultiHeadsClassifier
from data.data_utils import load_data
from train.trainer import GTrainer
from utils.dotdict import DotDict
from data.dataset import create_data_loader
# Pytorch
import torch
import torch.nn as nn
# utils
import argparse
from argparse import Namespace
import numpy as np
import random
from typing import Dict, Any, Union


def run(parameters: Union[DotDict, Namespace], mapping: Dict[str, Any]):
    # unpack parameters
    assert parameters.model_name_or_path is not None
    assert parameters.data_csv_path is not None
    output_dir = parameters.output_dir
    model_used = parameters.model_name_or_path
    data_csv_path = parameters.data_csv_path
    seed = parameters.random_seed
    lr = parameters.learning_rate
    max_len = parameters.max_len
    num_epochs = parameters.num_train_epochs
    batch_size = parameters.batch_size
    test_ratio = parameters.test_ratio
    warmup_steps = parameters.warmup_steps
    num_heads = parameters.num_heads
    device = torch.device("cuda:0") if \
        parameters.device == "gpu" and torch.cuda.is_available() else torch.device("cpu")

    # -----------------------------
    #     Fix the random seed
    #       Re-production
    # -----------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------------
    #   Loading the training data
    # -----------------------------
    # from_local the data & preprocess
    df, label_map, num_labels = load_data(data_csv_path, mapping, func_cleaning=None, placeholder="-",
                                          task_name="Read dataset...")

    # -----------------------------
    #      Model + Tokenizer
    # -----------------------------
    # create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_used, use_fast=True)
    # instantiate the model
    model = MultiHeadsClassifier(model_checkpoint=model_used, num_heads=num_heads, num_labels=num_labels)
    model.to(device)

    # -----------------------------
    #  Prepare Batched Dataloader
    # -----------------------------
    # create the train/val dataloader
    train_dataloader = create_data_loader(df, tokenizer, max_len=max_len, batch_size=batch_size)

    # -----------------------------
    #    Model Training Related
    # -----------------------------
    # optimizer
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    total_steps = len(train_dataloader) * num_epochs
    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # -----------------------------
    #        Training Time
    # -----------------------------
    # training via GTrainer
    trainer = GTrainer(model, tokenizer, device, optimizer, scheduler, loss_fn, max_len, num_heads, num_labels)
    trainer.train(num_epochs, train_dataloader, df)

    # save the model to output_dir
    trainer.save_model_tokenizer(output_dir, label_map)


if __name__ == "__main__":
    # -----------------------------
    #            configs
    # -----------------------------
    parser = argparse.ArgumentParser(description="train_classifier")
    parser.add_argument("--output_dir", type=str, default="./classifier_output")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--data_csv_path", type=str, default="/home/guo/df_sentiment_train.csv")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_ratio", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--device", type=str, default="gpu")

    params = parser.parse_args()

    # -----------------------------
    #            Entry
    # -----------------------------
    # specify how to use data for training
    dataMap = {
        # labels column names
        "labels": ["Topic Level1", "Topic Level2"],
        # text column name
        "text": "Body"
    }

    run(parameters=params, mapping=dataMap)
