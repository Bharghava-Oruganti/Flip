from imagefeatureextractor import *
from user_feature import *
from product_feature import *
from api import *
from imagefeatureextractor import ImageFeatureExtractor
import torch
import datetime
import torch.nn.functional as F
# # Example date strings in "yyyy-mm-dd" format
# # dates = ["2023-08-15", "2023-08-16", "2023-08-17"]

# # Convert date strings to datetime objects
# date_objects = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]

# # Extract year, month, and day components
# years = torch.tensor([date.year for date in date_objects])
# months = torch.tensor([date.month for date in date_objects])
# days = torch.tensor([date.day for date in date_objects])

# # One-hot encode year, month, and day
# year_encoded = torch.nn.functional.one_hot(years - years.min(), num_classes=10)  # Example: 10 years
# month_encoded = torch.nn.functional.one_hot(months - 1, num_classes=12)  # 12 months
# day_encoded = torch.nn.functional.one_hot(days - 1, num_classes=31)  # 31 days

# # Concatenate one-hot encoded vectors
# input_data = torch.cat((year_encoded, month_encoded, day_encoded), dim=1)

# print(input_data)
api = API()
def prodcut_attribute_to_feature(Id):
    try:
        attribute = api.product_data[Id]
    except KeyError:
        return torch.zeros(1000)
    extract = TextToFeature()
    print(attribute)
    image_extractor = ImageFeatureExtractor(list(attribute[0]["imageURL"][0]))
    print(attribute[0]["imageURL"][0])
    # word2v ;( not working
    required = extract.predict(attribute[0]["category"][0] + attribute[0]["description"][0] + attribute[0]["title"] +  attribute[0]["price"])
    # price = torch.tensor(attribute["price"])
    image = image_extractor.extract()
    return torch.cat((required, image))

def process_item():
    for ID, attribute in api.get_product_attribute():
        api.store_product_feature(ID, prodcut_attribute_to_feature(attribute))

def user_attribute_to_feature(attributes):
    # ret = []
    # for user in attributes:
    date_objects = [datetime.datetime.strptime(attribute["date"], "%Y-%m-%d") for attribute in attributes]
    
    years = torch.tensor([date.year for date in date_objects], dtype=torch.int32)
    months = torch.tensor([date.month for date in date_objects], dtype=torch.int32)
    days = torch.tensor([date.day for date in date_objects], dtype=torch.int32)
    print(years - years.min())

    # # years = torch.nn.functional.one_hot(years, num_classes = 10)
    # months = torch.nn.functional.one_hot(months - 1, num_classes = 12)
    # days = torch.nn.functional.one_hot(days - 1, num_classes = 31)
    
    print(years.shape)
    

    ratings = torch.tensor([[data["rating"]] for data in attributes], dtype = torch.float)
    reviews = torch.tensor([sentiment(data["reviewText"]) for data in attributes], dtype = torch.float)
    product_features = torch.tensor([[prodcut_attribute_to_feature(data["productId"])] for data in attributes], dtype = torch.float)
    return torch.cat((years.view(-1, 1), months.view(-1, 1), days.view(-1, 1), ratings, reviews, product_features))
        # for attribute in user:
        #     date_objects = [datetime.datetime.strptime(date["date"], "%Y-%m-%d") for date in attribute]
            
        #     years = torch.tensor([date.year for date in date_objects])
        #     months = torch.tensor([date.month for date in date_objects])
        #     days = torch.tensor([date.day for date in date_objects])

        #     years = torch.nn.functional.one_hot(years - years.min(), num_class = 10)
        #     months = torch.nn.functional.one_hot(months - 1, num_class = 12)
        #     days = torch.nn.functional.one_hot(days - 1, num_classes = 31)

        #     rating = torch.tensor(attributes[""])

def initialize_users(input_size, num_layers, hidden_size, encode_size, model_name, train_model = True):
    encoder = AutoEncoder(input_size = input_size, 
                          num_layers = num_layers, 
                          hidden_size = hidden_size, 
                          encode_size = encode_size, 
                          model_name = model_name)
    if (train_model):
        encoder.train_model()
    for Id, features in api.get_user_attributes():
        feature = user_attribute_to_feature(features)
        print(feature.shape)
        # api.store_user_feature(Id, feature)
def train_reccomendation():
    pass

def main():
    # control parameters
    initialize_users(input_size=1, num_layers=1, hidden_size=50, encode_size=20, model_name="what.pth", train_model=False)

    # turn product attribute to feature tensor
    # process_item()
    # turn user preference to tensor
    
    
    # load reccomendation engine
    # take input and run reccomendation engine

    pass


if __name__ == "__main__":
    main()