from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarise_text(text_string):
    return summarizer(text_string, max_length=130, min_length=30, do_sample=False, truncation=True)
    # return summarizer(text_string, truncation=True)
