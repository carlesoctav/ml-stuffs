import re
import unicodedata
from os import remove

from datasets import load_dataset
from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer

def cleaning_text(datasets):
    def text_cleaning(input_string: str):
        lowercase = input_string.lower()
        remove_accented = (
            unicodedata.normalize("NFKD", lowercase)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        remove_extra_whitespace = re.sub(r"^\s*|\s\s*", " ", remove_accented).strip()
        return remove_extra_whitespace

    def process_datasets(batch):
        return {
            "text": text_cleaning(batch["title"] + " " + batch["content"]),
            "label": batch["sentiment"],
        }
    datasets = datasets.map(
        process_datasets,
        remove_columns=[
            "id",
            "title",
            "content",
            "rating",
            "sentiment",
            "date",
            "name",
        ],
    )

    return datasets


def transform_tokenize(datasets, model_name, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, return_tensors="pt", max_length=max_length
        )

    return datasets.map(tokenize, batched=True, batch_size=16)


class DataFromJson(LightningDataModule):
    def __init__(
        self,
        data_path = "./data/raw/15239678.jsonl",
        model_name = "bert-base-uncased",
        batch_size=16,
        test_ratio=0.2,
        max_length=512,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.tokenize = transform_tokenize
        self.text_cleaning = cleaning_text
        self.max_length = max_length
        self.last_name = self.data_path.split("/")[-1].split(".")[0]


    def prepare_data(self):
        datasets = load_dataset("json", data_files=self.data_path)
        datasets = self.text_cleaning(datasets)
        datasets = self.tokenize(datasets, self.model_name, self.max_length)
        if self.test_ratio != 0:
            datasets = datasets["train"].train_test_split(test_size=self.test_ratio)
            train_ds = datasets["train"]
            eval_ds = datasets["test"]
            train_ds.to_json(f"./data/sent_classifier/{self.last_name}_train.jsonl")
            eval_ds.to_json(f"./data/sent_classifier/{self.last_name}_eval.jsonl")
        else:
            datasets["train"].to_json(f"./data/sent_classifier/{self.last_name}_train.jsonl")

    def setup(self, stage:str):
        if stage =="fit":
            if self.test_ratio != 0:
                self.train_ds = load_dataset("json", data_files=f"./data/sent_classifier/{self.last_name}_train.jsonl")
                self.eval_ds = load_dataset("json", data_files=f"./data/sent_classifier/{self.last_name}_eval.jsonl")
            else:
                self.train_ds = load_dataset("json", data_files=f"./data/sent_classifier/{self.last_name}_train.jsonl")
                self.eval_ds = load_dataset("json", data_files=f"./data/sent_classifier/{self.last_name}_train.jsonl")


            self.train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "token_type_ids"])
            self.eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "token_type_ids"]) 

    def train_dataloader(self):
        return DataLoader(self.train_ds["train"], batch_size=self.batch_size)

    def val_dataloader(self): 
        return DataLoader(self.eval_ds["train"], batch_size=self.batch_size)


if __name__ == "__main__":
    data_path = "./data/raw/15239678.jsonl"
    model_name = "bert-base-uncased"
    dm = DataFromJson(data_path, model_name)
    dm.prepare_data()
    dm.setup(stage = "fit")
    loader = dm.train_dataloader()
    print(next(iter(loader)))
