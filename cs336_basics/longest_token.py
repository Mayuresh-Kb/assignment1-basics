max_length = 0
longest_token = ""
longest_token_id = 0 

with open("../owt_32k_vocab.txt", "r", encoding='utf-8') as f:
    for line in f:
        token_id, token_hex = line.split()
        token = bytes.fromhex(token_hex)
        len_byte = len(token)
        if len_byte > max_length:
            max_length = len_byte
            longest_token = token
            longest_token_id = token_id

print(max_length, longest_token, longest_token_id)