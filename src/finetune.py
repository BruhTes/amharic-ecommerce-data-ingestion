# Amharic NER Fine-Tuning Script (Google Colab/Local GPU)
# Task 3: Fine-tune NER model on Amharic Telegram messages (Product, Price, Location)
# Compatible with 'Davlan/bert-tiny-amharic', 'xlm-roberta-base', or 'Davlan/afro-xlmr-mini'

# 1. Install Dependencies (Uncomment if in Colab)
# !pip install transformers datasets seqeval accelerate

import os
import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)

# 2. Parse Labeled CoNLL Data
def parse_conll_file(filepath):
    sentences, labels = [], []
    with open(filepath, encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    token, tag = splits
                    tokens.append(token)
                    tags.append(tag)
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

# === Edit this path to your labeled CoNLL file ===
conll_file = "./data/labeled_subset_conll.txt"
sentences, tags = parse_conll_file(conll_file)

# 3. Prepare DataFrame and Dataset
df = pd.DataFrame({'tokens': sentences, 'ner_tags': tags})
dataset = Dataset.from_pandas(df)

# 4. Train/Validation Split (20% validation)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

# 5. Label Mapping
label_list = sorted({label for doc in tags for label in doc})
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# 6. Load Tokenizer and Model
model_checkpoint = "Davlan/bert-tiny-amharic"  # Or "xlm-roberta-base", "Davlan/afro-xlmr-mini"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id
)

# 7. Tokenize and Align Labels
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"], is_split_into_words=True, truncation=True, padding='max_length', max_length=128
    )
    word_ids = tokenized_inputs.word_ids()
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label_to_id[example["ner_tags"][word_idx]])
        else:
            label_ids.append(label_to_id[example["ner_tags"][word_idx]])
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=False)
val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=False)

# 8. Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 9. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 10. Metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels, true_preds = [], []
    for pred, label in zip(preds, labels):
        tl, tp = [], []
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                tl.append(id_to_label[l_])
                tp.append(id_to_label[p_])
        true_labels.append(tl)
        true_preds.append(tp)
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 11. Trainer Setup & Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# 12. Evaluate and Save Model
results = trainer.evaluate()
print("Validation Results:", results)

model.save_pretrained("./finetuned-amharic-ner")
tokenizer.save_pretrained("./finetuned-amharic-ner")

# 13. Example Inference Function
def predict_ner(text):
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = inputs.word_ids()
    result = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and inputs["attention_mask"][0, idx]:
            token = tokens[word_idx]
            label = id_to_label[predictions[idx]]
            result.append((token, label))
    return result

# Example usage
example_text = "በአዲስ አበባ የቤት ሳጥን ዋጋ 250 ብር"
print(predict_ner(example_text))

# ---- END OF SCRIPT ----
# To use in Colab: upload your labeled_subset_conll.txt and run this script. Adjust paths as needed.