import regex as re
from .pretokenization_example import find_chunk_boundaries

INPUT_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path=INPUT_PATH, vocab_size=260, special_tokens=None):
    vocab = {}
    merges = []
    for i in range(256):
        vocab[i] = bytes([i])

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    for token in special_tokens:
        vocab[len(vocab)] = bytes(token.encode("UTF-8"))

    frequency_table = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pieces = [chunk]

            for special_token in special_tokens:
                new_pieces = []
                for piece in pieces:
                    split_pieces = piece.split(special_token) 
                    new_pieces.extend(split_pieces)
                pieces = new_pieces

            for piece in pieces:
                matches = re.finditer(PAT, piece)
                for match in matches:
                    match = match.group()
                    match = match.encode("UTF-8")

                    temp_list = []
                    for i in range(len(match)):
                        temp_list.append(bytes([match[i]]))
                    match = tuple(temp_list)
                    if match in frequency_table:
                        frequency_table[match] += 1
                    else:
                        frequency_table[match] = 1

    pair_frequency = {}
    pair_occurances = {}
    for key, value in frequency_table.items():
        for pair in zip(key[:-1], key[1:]):
            pair_frequency[pair] = pair_frequency.get(pair, 0) + value
            pair_occurances[pair] = pair_occurances.get(pair, set())
            pair_occurances[pair].add(key)

    while len(vocab) < vocab_size:

        max_pair = max(pair_frequency, key= lambda x: (pair_frequency[x], x))
        merges.append(max_pair)
        merged_pair = max_pair[0] + max_pair[1]
        vocab[len(vocab)] = merged_pair
        
        temp_pair_occurance = pair_occurances[max_pair].copy()

        for sequence in temp_pair_occurance:
            new_sequence = []
            position = 0
            while position < len(sequence):
                if position + 1 < len(sequence):
                    if(sequence[position], sequence[position + 1]) == max_pair:
                        merged_token = max_pair[0] + max_pair[1]
                        new_sequence.append(merged_token)
                        position += 2
                    else:
                        new_sequence.append(sequence[position])
                        position += 1
                else:
                    new_sequence.append(sequence[position])
                    position += 1
            new_sequence = tuple(new_sequence)
            frequency = frequency_table[sequence]

            for pair in zip(sequence[:-1], sequence[1:]):
                pair_frequency[pair] -= frequency

            for pair in set(zip(sequence[:-1], sequence[1:])):
                pair_occurances[pair].remove(sequence)
            
            for pair in zip(new_sequence[:-1], new_sequence[1:]):
                pair_frequency[pair] = pair_frequency.get(pair, 0) + frequency
                pair_occurances[pair] = pair_occurances.get(pair, set())
                pair_occurances[pair].add(new_sequence)

            frequency_table[new_sequence] = frequency_table.get(new_sequence, 0) + frequency
            del frequency_table[sequence]

    return(vocab, merges)
 
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens is None:
            self.special_tokens = ["<|endoftext|>"]
        for token in self.special_tokens:
            if token.encode("UTF-8") not in vocab.values():
                vocab[len(vocab)] = bytes(token.encode("UTF-8"))

        self.reverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))

        self.merges_rank = {}
        for index, value in enumerate(self.merges):
            self.merges_rank[value] = index

    def encode(self, text):

        escaped_tokens = []
        encoded_ids = []

        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)

        for special_token in sorted_tokens:
            escaped_tokens.append(re.escape(special_token))
        special_token_pattern = "|".join(escaped_tokens)
        text = re.split(f"({special_token_pattern})", text)

        for piece in text:
            if piece == '':
                pass
            elif piece in self.special_tokens:
                encoded_ids.append(self.reverse_vocab[piece.encode("UTF-8")])
            else:
                matches = re.finditer(PAT, piece)
                new_matches = []
                for match in matches:
                    match = match.group()
                    match = match.encode("UTF-8")
                    match = list(match)
                    new_matches.append(match)

                outer_list = []
                for inner_el in new_matches:
                    inner_list = []
                    for i in inner_el:
                        inner_list.append(bytes([i]))
                    outer_list.append(inner_list)

                for pre_token in outer_list:
                    while(True):
                        best_merge_rank = len(self.merges)
                        best_position = 0
                        best_pair = None
                        for i, pair in enumerate(zip(pre_token[:-1], pre_token[1:])):
                            if pair in self.merges_rank:
                                if self.merges_rank[pair] < best_merge_rank:
                                    best_merge_rank = self.merges_rank[pair]  
                                    best_position = i
                                    best_pair = pair
                        
                        if(best_pair is None):
                            break
                        else:
                            merged_pair = best_pair[0] + best_pair[1]
                            pre_token[best_position : best_position + 2] = [merged_pair]
                
                    for b in pre_token:
                        encoded_ids.append(self.reverse_vocab[b])

        return encoded_ids
    
    def decode(self, ids):
        byte_id = []
        for id in ids:
            byte_id.append(self.vocab[id])
        decoded_byte = (b"".join(byte_id)).decode("UTF-8", errors="replace")
        return decoded_byte
    
    def encode_iterable(self, iterable):
        for text in iterable:
            for id in self.encode(text):
                yield id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {} 
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_hex = line.split()
                vocab[int(id_hex[0])] = bytes.fromhex(id_hex[1])

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = []
            for line in f:
                byte_list = line.split()
                merges.append((bytes.fromhex(byte_list[0]), bytes.fromhex(byte_list[1])))

        return cls(vocab, merges, special_tokens=None)

def serialization_helper(vocab, merges, vocab_filepath, merges_filepath):
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        for id, token in vocab.items():
            f.write(str(id) + ' ' + token.hex() + "\n")

    with open(merges_filepath, "w", encoding="utf-8") as f:
        for first, second in merges:
            f.write(first.hex() + ' ' + second.hex() + "\n")


