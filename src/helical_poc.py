import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lstm")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    print(f"[helical_poc] OK: model={args.model} seq_len={args.seq_len} epochs={args.epochs}")

if __name__ == "__main__":
    main()
