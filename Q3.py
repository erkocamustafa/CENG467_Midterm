"""
CENG 467 - Q3: Text Summarization
Dataset: CNN/DailyMail
Models: TextRank (Extractive), DistilBART (Abstractive)
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import evaluate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Q3] Running on: {device}")

import nltk
nltk.download('punkt', quiet=True)

# 1. DATASET
print("Loading CNN/DailyMail dataset...")
# Using HF dataset to pull directly, keeping subset small for fast processing
dataset = load_dataset("cnn_dailymail", "3.0.0")
test_subset = dataset['test'].select(range(100)) # Evaluate on 100 articles

articles = [example['article'] for example in test_subset]
references = [example['highlights'] for example in test_subset]

# 2. EXTRACTIVE SUMMARIZATION (TextRank)
print("\n--- Running TextRank (Extractive) ---")
extractive_summaries = []
summarizer = TextRankSummarizer()

for text in articles:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Extract the top 3 sentences
    summary_sentences = summarizer(parser.document, 3)
    summary = " ".join([str(sentence) for sentence in summary_sentences])
    extractive_summaries.append(summary)

# 3. ABSTRACTIVE SUMMARIZATION (DistilBART)
print("\n--- Running DistilBART (Abstractive) ---")
model_name = "sshleifer/distilbart-cnn-12-6" # Lighter and faster than bart-large
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)

abstractive_summaries = []
model.eval()

with torch.no_grad():
    for article in articles:
        inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=30,
            num_beams=4,
            length_penalty=2.0
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        abstractive_summaries.append(summary)

# 4. EVALUATION METRICS
print("\n--- Computing Metrics ---")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

def print_metrics(preds, refs, model_type):
    # ROUGE
    r_scores = rouge.compute(predictions=preds, references=refs)
    # BLEU
    b_scores = bleu.compute(predictions=preds, references=[[r] for r in refs])
    # METEOR
    m_scores = meteor.compute(predictions=preds, references=refs)
    # BERTScore
    bs_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
    
    print(f"\n[{model_type} Results]")
    print(f"ROUGE-L:   {r_scores['rougeL']:.4f}")
    print(f"BLEU:      {b_scores['bleu']:.4f}")
    print(f"METEOR:    {m_scores['meteor']:.4f}")
    print(f"BERTScore: {np.mean(bs_scores['f1']):.4f}")

print_metrics(extractive_summaries, references, "TextRank (Extractive)")
print_metrics(abstractive_summaries, references, "DistilBART (Abstractive)")

# 5. QUALITATIVE EXAMPLE
print("\n" + "-"*50)
print("QUALITATIVE COMPARISON")
print("-"*50)
print(f"SOURCE ARTICLE:\n{articles[0][:300]}...\n")
print(f"REFERENCE:\n{references[0]}\n")
print(f"TEXTRANK (Extractive):\n{extractive_summaries[0]}\n")
print(f"DISTILBART (Abstractive):\n{abstractive_summaries[0]}\n")