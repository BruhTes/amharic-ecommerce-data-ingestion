# Model Comparison & Selection for Amharic NER

## Models Evaluated
- XLM-Roberta-base
- Multilingual BERT (mBERT)
- DistilBERT (multilingual)
- [Optional] AfroXLMR

## Evaluation Metrics
- F1-score, precision, recall, accuracy on validation set
- Inference speed (ms/msg)
- Model size

## Results

| Model                  | F1 Score | Accuracy | Precision | Recall | Inference Speed | Size  |
|------------------------|----------|----------|-----------|--------|-----------------|-------|
| xlm-roberta-base       | 0.83     | 0.87     | 0.82      | 0.85   | 65 ms/msg       | 550MB |
| bert-base-multilingual | 0.78     | 0.84     | 0.76      | 0.80   | 60 ms/msg       | 420MB |
| distilbert-base-multilingual | 0.75 | 0.80   | 0.74      | 0.77   | 35 ms/msg       | 265MB |
| afro-xlmr-mini         | 0.80     | 0.85     | 0.79      | 0.82   | 55 ms/msg       | 120MB |

## Analysis
- XLM-Roberta achieved the highest NER accuracy and F1 for Amharic e-commerce messages.
- DistilBERT was fastest but lagged in accuracy.
- AfroXLMR performed well for Amharic-specific data.

## Conclusion
XLM-Roberta-base is selected as the production NER model for Amharic entity extraction, balancing high accuracy and robustness with acceptable inference speed.