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


def _get_pretoken_counts(documents: list[str]) -> Counter[str]:
    pretoken_counts = Counter()

    for doc in documents:
        counts = _pretokenize_document(doc)

        pretoken_counts.update(counts)

    return pretoken_counts


def _pretokens_to_byte_sequence_counts(pretoken_counts: Counter[str]) -> Counter[tuple[bytes, ...]]:
    sequence_counts = Counter()

    for pretoken, count in pretoken_counts.items():
        encoded = pretoken.encode("utf-8")

        byte_seq = tuple(bytes([b]) for b in encoded)

        sequence_counts[byte_seq] = count

    return sequence_counts


def _initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    return vocab


def _count_adjacent_pairs(sequence_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    pair_counts = Counter()

    for seq, count in sequence_counts.items():
        for i in range(0, len(seq) - 1):
            pair = tuple(seq[i : i + 2])
            pair_counts[pair] += count

    return pair_counts


def _choose_best_pair(pair_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    _, best_count = pair_counts.most_common(1)[0]

    contenders = []

    for pair, count in pair_counts.items():
        if count == best_count:
            contenders.append(pair)

    best_pair = max(contenders)

    return best_pair


def _apply_merge(
    sequence_counts: Counter[tuple[bytes, ...]], best_pair: tuple[bytes, bytes], merged_token: bytes
) -> Counter[tuple[bytes, ...]]:
    updated_sequence_counts = Counter()

    for seq, count in sequence_counts.items():
        merged_seq = []

        i = 0

        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                merged_seq.append(merged_token)
                i += 2
            else:
                merged_seq.append(seq[i])
                i += 1

        merged_seq = tuple(merged_seq)

        updated_sequence_counts[merged_seq] += count

    return updated_sequence_counts


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    documents = _get_documents(corpus)

    pretoken_counts = _get_pretoken_counts(documents)

    sequence_counts = _pretokens_to_byte_sequence_counts(pretoken_counts)

    vocab = _initialize_vocab(special_tokens)

    merges = []

    while len(vocab) < vocab_size:
        pair_counts = _count_adjacent_pairs(sequence_counts)

        if not pair_counts:
            break

        best_pair = _choose_best_pair(pair_counts)

        merges.append(best_pair)

        merged_id = len(vocab)

        merged_token = best_pair[0] + best_pair[1]

        vocab[merged_id] = merged_token

        updated_sequence_counts = _apply_merge(sequence_counts, best_pair, merged_token)

        sequence_counts = updated_sequence_counts

    return vocab, merges
