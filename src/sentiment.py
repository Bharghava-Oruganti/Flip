from transformers import BertTokenizer, BertForSequenceClassification
import torch  
def sentiment(text):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)  
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)  
    outputs = model(**inputs)
    logits = outputs.logits  
    predicted_label = torch.argmax(logits, dim=1).item()  
    ret = torch.div(predicted_label, 4.)
    return ret