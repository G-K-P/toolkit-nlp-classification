from typing import Dict, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from data.dataset import create_data_loader
from utils.decorator import loading

NAN_PLACEHOLDER = "NOT_DEFINED"


def custom_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: customize this function and pass to load_data
    custom data cleaning function
    :param df: dataset to be cleaned
    :return: cleaned dataset
    """
    return df


def remove_private_info(df: pd.DataFrame, remove_email: bool = True, remove_time: bool = True,
                        remove_date: bool = True, remove_credit_card: bool = True, remove_street_address: bool = True,
                        remove_link: bool = True) -> pd.DataFrame:
    from commonregex import email, time, date, credit_card, street_address, link
    if remove_email:
        df = df.replace(email, "", regex=True)
    if remove_time:
        df = df.replace(time, "", regex=True)
    if remove_date:
        df = df.replace(date, "", regex=True)
    if remove_credit_card:
        df = df.replace(credit_card, "", regex=True)
    if remove_street_address:
        df = df.replace(street_address, "", regex=True)
    if remove_link:
        df = df.replace(link, "", regex=True)
    return df


def encode_labels(df, num_heads):
    labelEncoder = LabelEncoder()
    # encode the labels
    # level 1, no empty values
    label_mappings = {}
    for idx in range(num_heads):
        df.loc[:, f"label{idx + 1}"] = labelEncoder.fit_transform(df[f"label{idx + 1}"])
        label_mappings[f"label{idx + 1}"] = dict(zip(
            labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))

    # replace NaN with -1
    for idx in range(1, num_heads):
        if NAN_PLACEHOLDER in label_mappings[f"label{idx + 1}"]:
            df[f"label{idx + 1}"].replace(label_mappings[f"label{idx + 1}"][NAN_PLACEHOLDER], -1, inplace=True)

    num_labels = [df[f"label{idx + 1}"].max() + 1 for idx in range(num_heads)]

    # save the label map for later use
    label_map = {f"level{idx + 1}": {int(value): str(key) for key, value in label_mappings[f"label{idx + 1}"].items()}
                 for idx in range(num_heads)}
    return df, label_map, num_labels


@loading
def load_data(filename: str, mapping: Dict[str, str], func_cleaning: Callable, placeholder: str, **kwargs):
    """
    from_local dataset in .csv format, and perform data clean function on the dataset.
    Notice: 1) If there are more than one label, it will be considered as hierarchy classification dataset.
            So, the order of labels represents its position in the hierarchy.
            2) The first level's label is not nullable. If exists, rows will be filtered out
    :param kwargs: task_name argument for loading status (rich module)
    :param filename: path to the csv file
    :param mapping: dictionary to indicate the text column and labels columns
                    e.g.
                    {
                        "text": "Body",
                        "labels": ["label1", "label2"]
                    }
    :param func_cleaning: callable function to clean the text column.
    :param placeholder: blank cells' placeholder, if not defined, the empty cells already filled with numpy.NaN
    :return: Dataframe of the dataset
    """
    columns = [*mapping["labels"], mapping["text"]]
    num_heads = len(mapping["labels"])
    df = pd.read_csv(filename, usecols=columns)
    # rename the columns
    columns_dict = {
        **{val: f"label{index + 1}" for index, val in enumerate(mapping["labels"])},
        mapping["text"]: "text"
    }
    df.rename(columns=columns_dict, inplace=True)
    # replace placeholder string if needed
    df.replace(placeholder, np.NaN, inplace=True)
    # filter out null label in first level
    df = df[df["label1"].notnull()]
    # clean private information
    df = remove_private_info(df)
    # fill the NaN
    df.fillna(NAN_PLACEHOLDER, inplace=True)
    # encode the labels (if needed)
    assert isinstance(df["label1"][0], str)
    df, label_map, num_labels = encode_labels(df, num_heads=num_heads)

    # custom cleaning function
    if func_cleaning:
        df = func_cleaning(df)

    return df, label_map, num_labels


if __name__ == '__main__':
    filename_ = "../data_files/edcc_one_week.csv"
    mapping_ = {
        "text": "Body",
        "labels": ["Topic Level1", "Topic Level2"]
    }
    df_, label_map_, num_labels_ = load_data(filename_, mapping_, func_cleaning=custom_clean, placeholder="-",
                                             task_name="Read dataset...")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataloader = create_data_loader(df_, tokenizer, max_len=128, batch_size=16)
