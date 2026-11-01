import torch
from cs336_basics.transformer import Transformer_lm
from cs336_basics.tokenizer import Tokenizer

model = Transformer_lm(
    vocab_size = 10000,
    context_length = 128,
    d_model = 256,
    num_layers = 4,
    num_heads = 8,
    d_ff = 672,
    rope_theta = 10000.0
)

checkpoint = torch.load(
    "checkpoints/tinystories_continue_5000/checkpoint_5000.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model"])
model.eval()

tokenizer = Tokenizer.from_files("tinystories_10k_vocab.txt", "tinystories_10k_merges.txt")
prompt = input("Enter prompt: ")
encoded_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([encoded_ids], dtype=torch.long)
temperature = 0.7

def sample_next_token(logits, temperature):
    result = logits / temperature
    softmax_res = torch.softmax(result, dim=0)
    sampled_res = torch.multinomial(softmax_res, num_samples=1)
    return(sampled_res)

updated_ids = input_ids
for _ in range(256):
    acceptable_ids = updated_ids[:, -128:]
    with torch.no_grad():
        logits = model(acceptable_ids)
    next_token_logits = logits[0, -1, :]
    sampled_res = sample_next_token(next_token_logits, temperature)
    sampled_res = sampled_res.unsqueeze(0)
    if(sampled_res.squeeze()) == tokenizer.reverse_vocab["<|endoftext|>".encode("UTF-8")]:
        break
    updated_ids = torch.cat((updated_ids, sampled_res), dim=1)

    
updated_ids = updated_ids.squeeze().tolist()
decoded_ids = tokenizer.decode(updated_ids)
print(decoded_ids)