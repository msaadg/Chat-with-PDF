from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer.save_pretrained("models/distilbert-base-uncased-distilled-squad")
model.save_pretrained("models/distilbert-base-uncased-distilled-squad")
