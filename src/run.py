import imagefeatureextractor
from user_feature import *
from imagefeatureextractor import ImageFeatureExtractor

def process_item():
    extract = imagefeatureextractor.ImageFeatureExtractor()
    pass
def initialize_users(input_size, num_layers, hidden_size, encode_size, model_name, train_model = True):
    encoder = AutoEncoder(input_size = input_size, 
                          num_layers = num_layers, 
                          hidden_size = hidden_size, 
                          encode_size = encode_size, 
                          model_name = model_name)
    if (train_model):
        encoder.train_model()
    pass
def train_reccomendation():
    pass

def main():
    # control parameters


    # turn product attribute to feature tensor
    process_item()
    # turn user preference to tensor
    
    
    # load reccomendation engine
    # take input and run reccomendation engine

    pass


if __name__ == "__main__":
    main()