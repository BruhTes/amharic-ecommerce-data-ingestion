# Amharic E-commerce Data Ingestion and NER Labeling

This project is part of the B5W4 challenge: building an Amharic E-commerce Data Extractor for EthioMart.  
It includes a pipeline for:
- Ingesting and preprocessing data from Ethiopian Telegram e-commerce channels (Task 1)
- Manual NER labeling in CoNLL format (Task 2)

---

## Task 1: Data Ingestion and Preprocessing

**1. Environment Setup**
- Clone this repository.
- Create a virtual environment and activate it.
- Install requirements:
  ```bash
  pip install -r requirements.txt
  ```
- Register for Telegram API credentials at [my.telegram.org](https://my.telegram.org).
- Create a `.env` file in the root directory:
  ```
  TELEGRAM_API_ID=your_id
  TELEGRAM_API_HASH=your_hash
  TELEGRAM_SESSION_NAME=amharic_ecom_scraper
  ```

**2. Organize Data**
- Place the following in `data/`:
    - `channels_to_crawl.txt`: List of Telegram channels (one per line, no @)
    - `labeled_telegram_product_price_location.txt`: Provided NER data
    - `raw_telegram_data.csv`: Your output from data ingestion
    - `amharic_news_ner_train.txt`: Amharic NER data from [uhh-lt/ethiopicmodels](https://github.com/uhh-lt/ethiopicmodels/blob/main/am/data/NER/train.txt)
- Extract `photos.zip` into the `images/` directory.

**3. Ingest Telegram Data**
- Run the ingestion script:
  ```bash
  python src/telegram_ingest.py
  ```
- This fetches messages and images from selected Telegram channels and saves them to `data/raw_telegram_data.csv`, with images in `images/`.

**4. Preprocess Data**
- Run:
  ```bash
  python src/preprocess.py
  ```
- This cleans Amharic text, tokenizes, and produces `data/preprocessed_telegram_data.csv`.

---

## Task 2: Manual NER Labeling in CoNLL Format

**1. Manual Labeling**
- Run the script to annotate a subset of messages:
  ```bash
  python src/manual_label_conll.py
  ```
- Label each token as prompted:
    - B-Product, I-Product
    - B-LOC, I-LOC
    - B-PRICE, I-PRICE
    - O (for all non-entity tokens)
- Label at least 30-50 messages.

**2. Output**
- Labels are saved in `data/labeled_subset_conll.txt`, formatted as required for NER (CoNLL).

---

## Notes

- Use only your own API credentials, never share `.env` files publicly.
- All scripts are modular for easy extension and reproducibility.
- For a summary of your actual data, see the PDF report (to be completed after running the pipeline).

---