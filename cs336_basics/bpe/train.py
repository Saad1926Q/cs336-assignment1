from collections import Counter, defaultdict
from pathlib import Path

import regex as re

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_document(doc: str) -> Counter[str]:
    freq = Counter()

    for match in re.finditer(PATTERN, doc):
        text = match.group()

        freq[text] += 1

    return freq


def _get_pretoken_counts_streaming(input_path: str) -> Counter[str]:
    pretoken_counts = Counter()

    leftover = ""

    delimiter = "<|endoftext|>"

    chunk_size = 16 * 1024 * 1024

    with open(input_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            text = leftover + chunk

            parts = text.split(delimiter)

            for doc in parts[:-1]:
                counts = _pretokenize_document(doc)
                pretoken_counts.update(counts)

            leftover = parts[-1]

    if leftover:
        pretoken_counts.update(_pretokenize_document(leftover))

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


def _build_pair_counts_and_mappings(
    sequence_counts: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts = Counter()
    pair_to_sequences = defaultdict(set)

    for seq, count in sequence_counts.items():
        for i in range(0, len(seq) - 1):
            pair = tuple(seq[i : i + 2])
            pair_counts[pair] += count
            pair_to_sequences[pair].add(seq)

    return pair_counts, pair_to_sequences


def _choose_best_pair(pair_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    _, best_count = pair_counts.most_common(1)[0]

    contenders = []

    for pair, count in pair_counts.items():
        if count == best_count:
            contenders.append(pair)

    best_pair = max(contenders)

    return best_pair


def _merge_sequence(
    sequence: tuple[bytes, ...],
    best_pair: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    merged_seq = []

    i = 0

    while i < len(sequence):
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == best_pair:
            merged_seq.append(merged_token)
            i += 2
        else:
            merged_seq.append(sequence[i])
            i += 1

    merged_seq = tuple(merged_seq)

    return merged_seq


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pretoken_counts = _get_pretoken_counts_streaming(input_path)

    sequence_counts = _pretokens_to_byte_sequence_counts(pretoken_counts)

    vocab = _initialize_vocab(special_tokens)

    merges = []

    pair_counts, pair_to_sequences = _build_pair_counts_and_mappings(sequence_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = _choose_best_pair(pair_counts)

        merges.append(best_pair)

        merged_id = len(vocab)

        merged_token = best_pair[0] + best_pair[1]

        vocab[merged_id] = merged_token

        affected_sequences = pair_to_sequences[best_pair].copy()

        for old_seq in affected_sequences:
            count = sequence_counts[old_seq]

            old_pairs_seen = set()

            for i in range(0, len(old_seq) - 1):
                pair = (old_seq[i], old_seq[i + 1])

                pair_counts[pair] -= count

                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                old_pairs_seen.add(pair)

            for pair in old_pairs_seen:
                pair_to_sequences[pair].remove(old_seq)

                if not pair_to_sequences[pair]:
                    del pair_to_sequences[pair]

            del sequence_counts[old_seq]

            new_seq = _merge_sequence(old_seq, best_pair, merged_token)

            sequence_counts[new_seq] += count

            for i in range(0, len(new_seq) - 1):
                pair = (new_seq[i], new_seq[i + 1])

                pair_counts[pair] += count
                pair_to_sequences[pair].add(new_seq)

    return vocab, merges
