from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class Embedder(ABC):
    """
    Abstract base class for embedding images into feature vectors.
    """

    @abstractmethod
    def embed(self, image: Image.Image) -> np.ndarray:
        """
        Abstract method to embed an image into a feature vector.
        """
        pass


class ResNet50Embedder(Embedder):
    """
    Embedder class that uses a pre-trained ResNet50 model to generate image embeddings.
    """

    def __init__(self) -> None:
        """
        Initializes the ResNet50 embedder by setting up the image preprocessing pipeline
        and loading the pre-trained ResNet50 model with the final classification layer removed.
        """
        super().__init__()

        # Define image preprocessing pipeline.
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the pre-trained ResNet50 model and remove the final classification layer.
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        """
        Embeds an image into a feature vector using the ResNet50 model.
        """
        # Preprocess the image.
        processed_image = self.preprocessor(image).unsqueeze(0)

        # Generate the embedding without tracking gradients.
        with torch.no_grad():
            embedding = self.model(processed_image)

        # Squeeze the result to remove unnecessary dimensions and convert to a NumPy array.
        return embedding.squeeze().numpy()
