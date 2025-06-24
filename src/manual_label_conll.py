import pandas as pd
import os

DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_telegram_data.csv'))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled_subset_conll.txt'))

LABELS = [
    "O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"
]

def label_message(tokens):
    labels = []
    print("\nLabel tokens (type the label or just enter for 'O'):")
    for token in tokens:
        while True:
            label = input(f"{token}: ").strip()
            if label == "": label = "O"
            if label in LABELS:
                labels.append(label)
                break
            else:
                print(f"Invalid label! Use one of: {LABELS}")
    return labels

def main():
    df = pd.read_csv(DATA_FILE)
    df = df.sample(n=40, random_state=42)  # Pick 40 random messages

    all_labeled = []
    for idx, row in df.iterrows():
        print("\n" + "="*50)
        print(f"Message: {row['clean_text']}")
        tokens = eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
        print("Tokens:", tokens)
        labels = label_message(tokens)
        for token, label in zip(tokens, labels):
            all_labeled.append(f"{token} {label}")
        all_labeled.append("")  # Blank line to separate messages

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_labeled))
    print(f"\nLabeled data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()