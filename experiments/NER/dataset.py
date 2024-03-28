from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from datasets import Dataset, DatasetDict, load_dataset
from lightning import LightningDataModule
from transformers import AutoTokenizer, BertTokenizer

def tokenize_and_align_labels(tokenizer, ds:Dataset, max_length):
    def do_the_jobs(batch):
        tokenized_input = tokenizer(batch["tokens"], is_split_into_words=True, padding="max_length", truncation=True, max_length=max_length)
        labels = []
        for idx, label in enumerate(batch["ner_tags"]):
            word_ids = tokenized_input.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_input["labels"] = labels
        return tokenized_input

    return ds.map(do_the_jobs, batched = True, remove_columns = ["langs", "ner_tags", "tokens"])


@dataclass
class PanXDataset(LightningDataModule):
    langs: List[str]
    tokenizer_name: str
    props: Optional[List[float]] = None
    seed: int = 42
    max_length: Optional[int] = None

    def prepare_data(self):
        self.ch = defaultdict(DatasetDict)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if self.max_length is None:
            self.max_length = tokenizer.model_max_length 

        if self.props is None:
            self.props = [1/len(self.langs) for _ in self.langs]

        if sum(self.props) != 1:
            raise ValueError("The sum of the proportions must be 1")

        for lang, prop in zip(self.langs, self.props):
            ds = load_dataset("xtreme", name = f"PAN-X.{lang}")
            for split in ds:
                if not isinstance(ds, DatasetDict):
                    raise ValueError("The dataset must be a DatasetDict")

                self.ch[lang][split] = ds[split].shuffle(seed = self.seed).select(range(int(prop * ds[split].num_rows)))
                self.ch[lang][split] = tokenize_and_align_labels(tokenizer, self.ch[lang][split], self.max_length)


        print(f"DEBUGPRINT[1]: dataset.py:60: self.ch={self.ch}")

    def get_example(self, lang:str, num_example:int = 2):
        tags = self.ch[lang]["train"].features["ner_tags"].feature
        print(f"DEBUGPRINT[3]: dataset.py:35: tags={tags}")
        return [(self.ch[lang]["train"][i]["tokens"], tags.int2str(self.ch[lang]["train"][i]["ner_tags"])) for i in range(num_example)]

    def setup(self, stage=None):
        pass
    def train_dataloader(self):
        ...

def main():
    langs = ["id", "en"]
    ds = PanXDataset(langs, "nreimers/mmarco-mMiniLMv2-L12-H384-v1")
    ds.prepare_data()
    a = ds.ch["id"]["train"][:2]
    print(f"DEBUGPRINT[2]: dataset.py:77: a={a}")


if __name__== "__main__":
    main()

