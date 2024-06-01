# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("gechim/xlm-roberta-base-finetuned_60kURL")