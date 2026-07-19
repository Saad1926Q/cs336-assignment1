from collections import Counter
from pathlib import Path

import regex as re

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _get_documents(corpus: str) -> list[str]:
    return corpus.split("<|endoftext|>")


def _pretokenize_document(doc: str) -> Counter[str]:
    freq = Counter()

    for match in re.finditer(PATTERN, doc):
        text = match.group()

        freq[text] += 1

    return freq


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    frequencies = Counter()

    documents = _get_documents(corpus)

    for doc in documents:
        counts = _pretokenize_document(doc)

        frequencies.update(counts)

    pretoken_byte_sequences = Counter()

    for pretoken, freq in frequencies.items():
        encoded = pretoken.encode("utf-8")

        byte_seq = tuple(bytes([b]) for b in encoded)

        pretoken_byte_sequences[byte_seq] = freq

    vocab = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    merges = []

    while len(vocab) < vocab_size:
        freq_counter = Counter()

        for seq, count in pretoken_byte_sequences.items():
            for i in range(0, len(seq) - 1):
                pair = tuple(seq[i : i + 2])
                freq_counter[pair] += count

        if not freq_counter:
            break

        _, best_freq = freq_counter.most_common(1)[0]

        contenders = []

        for pair, count in freq_counter.items():
            if count == best_freq:
                contenders.append(pair)

        best_pair = max(contenders)

        merges.append(best_pair)

        merged_id = len(vocab)

        merged_pair = best_pair[0] + best_pair[1]

        vocab[merged_id] = merged_pair

        new_seq = Counter()

        for seq, count in pretoken_byte_sequences.items():
            merged_seq = []

            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    merged_seq.append(merged_pair)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1

            merged_seq = tuple(merged_seq)

            new_seq[merged_seq] += count

        pretoken_byte_sequences = new_seq

    return vocab, merges
