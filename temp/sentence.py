from transformers import AutoTokenizer

def main():
    xlmr_model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
    output = tokenizer("i love food")
    print(f"DEBUGPRINT[1]: sentence.py:6: output={output}")
    token = output.tokens()
    print(f"DEBUGPRINT[5]: sentence.py:8: token={token}")

if __name__ == "__main__":
    main()
