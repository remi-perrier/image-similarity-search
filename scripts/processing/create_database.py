from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from lydia_image_similarity.database import NumpyVectorDatabase
from lydia_image_similarity.embedder import ResNet50Embedder

INDEX_FILEPATH = "data/index.csv"
DATABASE_SAVE_FILE = "data/database.npz"
IMAGES_FOLDER = "data/images"

index = pd.read_csv(INDEX_FILEPATH)

database = NumpyVectorDatabase(vector_size=2048)
embedder = ResNet50Embedder()


for _, row in tqdm(index.iterrows(), total=len(index)):
    image_id = row["id"]
    image_url = row["url"]
    image_filepath = Path(IMAGES_FOLDER, image_url.split("/")[-1])

    try:
        image = Image.open(image_filepath)
        image_vector = embedder.embed(image)
        database.add(image_vector, image_id, {"filepath": str(image_filepath)})
    except Exception as e:
        print(image_id, e)

database.save(DATABASE_SAVE_FILE)
