from torch import isin
from layers.transformers_hf import XLMRobertaForTokenClassification
from transformers import AutoTokenizer, AutoConfig, RobertaModel, XLMConfig, XLMRobertaConfig, Trainer

def main():
    config = AutoConfig.from_pretrained("nreimers/mmarco-mMiniLMv2-L12-H384-v1", num_labels = 10) 
    model= XLMRobertaForTokenClassification.from_pretrained("nreimers/mmarco-mMiniLMv2-L12-H384-v1", config = config)

    if not isinstance(model, XLMRobertaForTokenClassification):
        return False

    tokenizer = AutoTokenizer.from_pretrained("nreimers/mmarco-mMiniLMv2-L12-H384-v1")
    input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = model(**input)
    print(f"DEBUGPRINT[2]: roberta.py:9: input={input}")
    print(f"DEBUGPRINT[6]: roberta.py:14: output={output}")
    print(output["logits"].shape)


if __name__ == "__main__":
    main()
