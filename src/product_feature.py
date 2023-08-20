from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch  
import torch.functional as F
from img2vec_pytorch import Img2Vec
from PIL import Image
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


# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ImageFeatureExtractor():
      # Initialize Img2Vec with GPU
       def __init__(self, files):
            self.img2vec = Img2Vec(model='resnet-18', cuda=cuda_available)

            self.files = files

       def extract(self):

            # Read in an image (rgb format)
            list_images = []
            for img in self.files:
                image = Image.open(img)
                new_size = (224, 224)  # Replace with the desired dimensions

                # Resize the image
                image = image.resize(new_size)

                # plt.imshow(image)
                # plt.axis('off')  # Turn off axis labels and ticks
                # plt.show()
                list_images.append(image)

            # img = Image.open('/content/gal1.jpg')
            # Get a vector from img2vec, returned as a torch FloatTensor
            features = self.img2vec.get_vec(list_images, tensor=True)
            # Or submit a list
            # vectors = img2vec.get_vec(list_of_PIL_images)
            print(features)
            print(features.shape)

            flattened_features = torch.flatten(features, start_dim=1)
            print(flattened_features.shape)
            return flattened_features
