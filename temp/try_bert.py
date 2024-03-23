from transformers import BertModel, BertTokenizer

def main():
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inp = ["saya suka makan nasi", "apple tree"]
    inp_tensor = tokenizer(inp, return_tensors='pt', padding=True, truncation=True)
    print(f"DEBUGPRINT[1]: try_bert.py:7: inp_tensor={inp_tensor}")
    out = model(**inp_tensor)
    print(f"DEBUGPRINT[2]: try_bert.py:9: out={out}")

    for x in out.values():
        print(f"DEBUGPRINT[3]: try_bert.py:12: x={x.shape}")

if __name__ == '__main__':
    main()
