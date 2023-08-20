from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch  
import torch.functional as F
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
def discription_feature(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    input_data = "This is a product description for a laptop. It has a 15.6-inch screen, an Intel Core i5 processor, and 8GB of RAM."
    
    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)
    
    in_1 = "HP laptop"
    in_2 = "Clay Pot"
    input_1 = tokenizer(in_1, return_tensors = "pt", padding = True, truncation = True)
    input_2 = tokenizer(in_2, return_tensors = "pt", padding = True, truncation = True)
    
    with torch.no_grad():
        out_1 = model(**input_1)
    with torch.no_grad():
        out_2 = model(**input_2)
    
    
    print(out_1.last_hidden_state.shape)
    print(out_2.last_hidden_state.shape)
    print(F.cosine_similarity(out_1.last_hidden_state[:, -1, :], out_2.last_hidden_state[:, -1, :]))
class TextToFeature():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.next_model = self.model.clone()
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs[:, -1, :]
    def train(self, first_data, second_data, num_epochs):
        
        pass