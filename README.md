# CENG 467: Natural Language Understanding and Generation
**Take-Home Midterm Project**  
**Izmir Institute of Technology (IZTECH) - Computer Engineering**

**Student:** Mustafa ERKOCA (300201052)  
**Instructor:** Prof. Dr. Aytuğ Onan

---

## 📌 Project Overview
This repository contains the implementation and analysis of five core Natural Language Processing (NLP) tasks as part of the CENG 467 Midterm assignment. The project explores various paradigms ranging from classical statistical models to state-of-the-art contextual transformers.

### 📝 Tasks Breakdown
*   **[Q1] Text Classification via Representation Learning:** Sentiment analysis on the IMDb dataset comparing sparse (TF-IDF + SVM) and contextual (`distilbert-base-uncased`) representations.
*   **[Q2] Named Entity Recognition (NER):** Sequence labeling on the CoNLL-2003 dataset. Compares a classical `BiLSTM-CRF` (trained from scratch) against a pre-trained `distilbert-base-cased` model, emphasizing the impact of contextual embeddings and subword token alignment.
*   **[Q3] Text Summarization:** Evaluates extractive (`TextRank`) vs. abstractive (`sshleifer/distilbart-cnn-12-6`) summarization paradigms on the CNN/DailyMail dataset.
*   **[Q4] Machine Translation:** Seq2Seq translation (German to English) on the Multi30k dataset. Explores custom Bahdanau Attention mechanisms vs. pre-trained Transformer baselines (`Helsinki-NLP/opus-mt-de-en`).
*   **[Q5] Language Modeling:** Compares a statistical N-gram (Trigram with Laplace smoothing) model against a neural LSTM architecture with weight tying for text generation and perplexity reduction.

---

## 📂 Repository Structure
```text
CENG467_Midterm/
│
├── datasets/            # Local datasets (Excluded from version control if >100MB)
├── q1.py                # Source code for Text Classification
├── q2.py                # Source code for Named Entity Recognition (NER)
├── q3.py                # Source code for Text Summarization
├── q4.ipynb             # Jupyter Notebook for Machine Translation
└── q5.py                # Source code for Language Modeling
```

---

## 🚀 Installation & Requirements
To run the scripts in this repository, ensure you have Python 3.8+ installed. 

Install the required dependencies using pip:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers seqeval scikit-learn numpy spacy pytorch-crf
```

**⚠️ Important Note for Q2 (NER):** 
The BiLSTM-CRF implementation requires the `pytorch-crf` library. Do NOT install `TorchCRF` as it may cause version conflicts. If you have conflicting packages, clean them up first:
```bash
pip uninstall TorchCRF torchcrf -y
pip install pytorch-crf
```

---

## ⚙️ How to Run
You can execute the Python scripts directly from the terminal. For example, to run the NER pipeline:
```bash
python q2.py
```

For Q4, open the Jupyter Notebook to run the cells interactively:

```bash
jupyter notebook q4.ipynb
```
