import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from lydia_image_similarity.database import NumpyVectorDatabase
from lydia_image_similarity.embedder import ResNet50Embedder

DATABASE_FILEPATH = "data/database.npz"

database = NumpyVectorDatabase(vector_size=2048)
database.load(DATABASE_FILEPATH)

embedder = ResNet50Embedder()

app = FastAPI(title="Image Similarity Search by Rémi PERRIER")


@app.post("/similarity-search/")
async def upload_image(file: UploadFile = File(...)):
    # Ensure that the uploaded file is an image.
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image"}, status_code=400)

    # Read the file and convert it to an image.
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))

    except Exception:
        return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

    # Embed the image.
    image_vector = embedder.embed(image)

    # Found images with greatest similarity from database.
    top_ids = database.top_k_similarities(image_vector)

    return {"ids": top_ids}
