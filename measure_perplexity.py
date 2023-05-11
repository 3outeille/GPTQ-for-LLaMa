
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Model
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained("sgugger/rwkv-430M-pile", torch_dtype=torch.float16, device_map="auto")
print(model.hf_device_map)
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

# Dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
ctx_len = 1024
stride = ctx_len // 2
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
# for begin_loc in tqdm(range(0, seq_len, stride)):
for begin_loc in tqdm(range(0, stride*20, stride)):
    end_loc = min(begin_loc + ctx_len, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(f"Perplexity: {ppl}")