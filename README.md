# Image Similarity Search

## Project Structure

The project is composed of 4 sub-folders :

- `data` : the folder containing the CSV file from the assessment, and where all data will be saved.  
- `notebooks` : contains two notebooks : a quick EDA of the images, and a test of the approach for similarity search.  
- `scripts` : contains the scripts for the preprocessing, and the API endpoint.  
- `src/lydia_image_similarity` : the core package, implementing an embedder and a vector database.  

In addition, the project contains a few files :

- `Dockerfile` and `.dockerignore` : allow to run the API in a container.  
- `pyproject.toml` and `poetry.lock` : the project information and the python environment managed using poetry.
- `Makefile` : a few commands to make it easier to start the scripts.
- `.gitignore` : ignore all data except the index, that will be kept by simplicity.  


## How to use the code

### Docker

```
docker run -p 8000:8000 --rm -it $(docker build -q .)
``` 
The command will create a container exposing the API on the host machine. The API documentation will be available at http://localhost:8000/docs.

### Manual

Clone the project locally. 

Inside the project root folder, 

```
poetry install
```

will create the Python environment.  

#### Without make

```
# Download the images.
mkdir data/images 
poetry run python scripts/processing/download.py

# Create the vector database.
poetry run python scripts/processing/create_database.py

# Start the API in prod mode.
poetry run fastapi run scripts/api/app.py

# Or start the API in prod mode.
poetry run fastapi dev scripts/api/app.py
```

#### Using make

If you have make installed locally, you can simply run 

```
make run-api-prod
```

to execute the different scripts in order.


### API

The API endpoint to perform similarity search is (POST) `/similarity-search`.  
The answer format is 

```
{
    "ids":[]
}
```


### Explanation

#### Introduction

Image similarity search is a common problem in many computer vision applications, such as content-based image retrieval, recommendation systems, and visual search engines. This project implements a simple yet effective image similarity search using vector embeddings and a custom API built with FastAPI.

Given the small dataset size, the system performs an exhaustive search by computing distances between the input image and all images in the dataset, ensuring exact matches. For larger datasets, a more scalable method like Approximate Nearest Neighbor (ANN) algorithms would be necessary to ensure fast retrieval times.

#### Embedding with ResNet50

We leverage a pre-trained ResNet50 model to transform images into high-dimensional vectors (embeddings). ResNet50 is a general-purpose deep neural network, widely used for image classification tasks. In our case:

- We remove the final classification layer to output a meaningful embedding vector that represents the image.
- The ResNet50 model is well-suited for this project because it works with a variety of image types, making it flexible for our diverse dataset.

#### Image Search with Vector Distance

Once images are embedded as vectors, the similarity search is performed by calculating the distance between the input image's vector and all vectors in the dataset. The Cosine Similarity is used as a measure of similarity (other distance metrics like Euclidean distance can also be explored).

For small datasets, this brute-force comparison works efficiently. For larger datasets, the current setup can be extended by implementing Approximate Nearest Neighbor (ANN) algorithms like FAISS or ScaNN for faster, scalable searches with trade-offs in exactness.

#### API Implementation

The API is built using FastAPI, which allows for:

- Quick development and deployment of a RESTful API.
- Auto-generated interactive API documentation with Swagger UI, making it easy to test endpoints during development.
