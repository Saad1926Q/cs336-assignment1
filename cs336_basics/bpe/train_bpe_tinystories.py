import pickle
import time

from cs336_basics.bpe.train import train_bpe

start = time.perf_counter()


def main():
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"]
    )

    elapsed = time.perf_counter() - start

    longest_token = max(vocab.values(), key=lambda token: len(token))

    with open("data/tinystories_bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    print(f"Training took {elapsed:.2f} seconds")
    print(f"Training took {elapsed / 60:.2f} minutes")

    print(f"Longest token is {longest_token!r}")
    print(f"Length of longest token is {len(longest_token)}")


if __name__ == "__main__":
    main()
