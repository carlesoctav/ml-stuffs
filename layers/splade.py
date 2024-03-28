from dataclasses import dataclass
from transformers import BertConfig, BertForMaskedLM, BertModel, BertPreTrainedModel, PreTrainedModel, BertTokenizer
from torch import nn
import torch


class BertSpalde(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config,add_pooling_layer=False)
        self.embed = self.bert.embeddings.word_embeddings.weight
        self.transform = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gelu = nn.GELU()
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        ...


def main():
    config = BertConfig.from_pretrained("bert-base-uncased")
    splade = BertSpalde.from_pretrained("bert-base-uncased", config=config) 

    if not isinstance(splade, BertSpalde):
        raise ValueError("splade is not an instance of BertSpalde")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inp = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print(f"DEBUGPRINT[2]: splade.py:31: inp={inp}")
    out = splade(**inp)
    print(f"DEBUGPRINT[1]: splade.py:27: out={out}")


if __name__ == "__main__":
    main()
    
