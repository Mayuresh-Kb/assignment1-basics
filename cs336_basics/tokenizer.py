import regex as re
from .pretokenization_example import find_chunk_boundaries

INPUT_PATH = "/Users/mayureshkasabe/Desktop/CS336-Language-Modeling/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
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
 
