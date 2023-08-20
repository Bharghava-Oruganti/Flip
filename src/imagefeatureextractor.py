from img2vec_pytorch import Img2Vec
from PIL import Image
import torch

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
