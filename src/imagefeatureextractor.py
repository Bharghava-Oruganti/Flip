from img2vec_pytorch import Img2Vec
from PIL import Image
import torch
from matplotlib import pyplot as plt
import urllib

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
                url, filename = (img, "download.jpg")
                try: urllib.URLopener().retrieve(url, filename)
                except: urllib.request.urlretrieve(url, filename)

                image = Image.open(filename)
                new_size = (224, 224)  # Replace with the desired dimensions

                # Resize the image
                image = image.resize(new_size)

                # Replace with your image path
                # downloaded_image = Image.open('downloaded_image.jpg')


                plt.imshow(image)
                plt.axis('off')  # Turn off axis labels and ticks
                plt.show()
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


def image_features(Image):
    feature_model = ImageFeatureExtractor([Image])
    # image url
    image_features = feature_model.extract()

    return image_features


# obj = {}
# obj['Image'] = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpw-3S_E2XvBAMXQxLeT_HoCJiYdG4WXFZTg&usqp=CAU'
# image_features(obj['Image'])
