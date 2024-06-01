# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("gechim/phobert-base-v2-finetuned_60kURL")
model = AutoModelForSequenceClassification.from_pretrained("gechim/phobert-base-v2-finetuned_60kURL")