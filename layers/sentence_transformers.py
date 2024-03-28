import os
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from datasets.arrow_dataset import query_table
from torch import nn
from transformers import (
    AutoConfig,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import DataLoader


class BertForEmbedding(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, embed_pooling: str = "cls"):
        super().__init__(config)
        self.bert = BertModel(config=config)
        self.embed_pooling = embed_pooling
        self.init_weights()

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self._pooling(output, attention_mask)
    def _pooling(self, output, attention_mask):
        if self.embed_pooling == "mean":
            return self._mean_pooling(output, attention_mask)
        elif self.embed_pooling == "cls":
            return self._cls_pooling(output)

    def _mean_pooling(self, output, attention_mask):
        token_embeddings = output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _cls_pooling(self, output):
        return output.last_hidden_state[:, 0, :]


def simpleCollator(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}


class TripletWithInbatchTrainer(Trainer):
    def __init__(self, *args, loss_function: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(self, model: BertForEmbedding, inputs, return_outputs=False):

        query_embed = model(
            input_ids=inputs["input_ids_query"],
            attention_mask=inputs["attention_mask_query"],
            token_type_ids=inputs["token_type_ids_query"],
        )
        pos_embed = model(
            input_ids=inputs["input_ids_positive"],
            attention_mask=inputs["attention_mask_positive"],
            token_type_ids=inputs["token_type_ids_positive"],
        )
        neg_embed = model(
            input_ids=inputs["input_ids_negative"],
            attention_mask=inputs["attention_mask_negative"],
            token_type_ids=inputs["token_type_ids_negative"],
        )

        pos_score = query_embed @ pos_embed.T  # [batch_size, batch_size]
        neg_score = torch.diag(query_embed @ neg_embed.T).unsqueeze(
            -1
        )  # [batch_size, 1]
        print(
            f"DEBUGPRINT[4]: sentence_transformers.py:74: neg_score.shape = {neg_score.shape}"
        )
        pos_score = torch.cat([pos_score, neg_score], dim=-1)
        print(f"DEBUGPRINT[2]: sentence_transformers.py:92: pos_score={pos_score.shape}")
        labels = torch.cat(
            (
                torch.diag(torch.ones(query_embed.shape[0])),
                torch.zeros(query_embed.shape[0], 1)
            ),
            dim=-1,
        )
        print(f"DEBUGPRINT[1]: sentence_transformers.py:92: labels={labels}")
        print(pos_score.shape)
        print(labels.shape)
        loss = nn.CrossEntropyLoss()(pos_score, labels)
        print(loss)
        return (loss,) if return_outputs else loss


def tokenize(datasets, tokenizer):
    def do_the_jobs(batch):
        query_token = tokenizer(
            batch["query"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        pos_token = tokenizer(
            batch["positive"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        neg_token = tokenizer(
            batch["negative"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        output_1 = {f"{k}_query": v for k, v in query_token.items()}
        output_2 = {f"{k}_positive": v for k, v in pos_token.items()}
        output_3 = {f"{k}_negative": v for k, v in neg_token.items()}
        output = {**output_1, **output_2, **output_3}
        return output

    return datasets.map(
        do_the_jobs,
        batched=True,
        num_proc=4,
        remove_columns=["query", "positive", "negative"],
    )


def main2():
    datasets = load_dataset("carles-undergrad-thesis/indo-mmarco-500k")
    if not isinstance(datasets, DatasetDict):
        raise ValueError("datasets is not an instance of dict")

    model = BertForEmbedding.from_pretrained("google-bert/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    datasets = datasets["train"].train_test_split(test_size=0.01)
    datasets = tokenize(datasets["test"], tokenizer)
    datasets.set_format(type="torch", columns=datasets.column_names)

    print(f"DEBUGPRINT[3]: sentence_transformers.py:79: datasets={datasets}")

    trainArgs = TrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        remove_unused_columns=False,
    )

    trainer = TripletWithInbatchTrainer(
        model=model,
        args=trainArgs,
        train_dataset=datasets,
        eval_dataset=datasets,
        loss_function=nn.TripletMarginLoss(),
        tokenizer=tokenizer,
        data_collator=simpleCollator,
    )

    trainer.train()


def main():
    model = BertForEmbedding.from_pretrained("google-bert/bert-base-uncased")
    if not isinstance(model, BertForEmbedding):
        raise ValueError("model is not an instance of BertForEmbedding")

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    inp = ["Hello, my dog is cute", "I love my dog"]
    inp = tokenizer(inp, return_tensors="pt", padding=True, truncation=True)
    print(f"DEBUGPRINT[2]: sentence_transformers.py:19: inp={inp}")
    output = model(**inp)
    print(f"DEBUGPRINT[2]: sentence_transformers.py:46: output={output}")


if __name__ == "__main__":
    main2()
