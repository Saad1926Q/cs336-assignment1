import pickle
import time

from cs336_basics.bpe.train import train_bpe

start = time.perf_counter()


def main():
    vocab, merges = train_bpe(input_path="data/owt_train.txt", vocab_size=32000, special_tokens=["<|endoftext|>"])

    elapsed = time.perf_counter() - start

    longest_token = max(vocab.values(), key=lambda token: len(token))

    with open("data/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("data/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print(f"Training took {elapsed:.2f} seconds")
    print(f"Training took {elapsed / 60:.2f} minutes")

    print(f"Longest token is {longest_token!r}")
    print(f"Length of longest token is {len(longest_token)}")


if __name__ == "__main__":
    main()
