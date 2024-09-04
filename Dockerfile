# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install Poetry and other dependencies
RUN pip install --upgrade pip --no-cache-dir && \
    pip install poetry --no-cache-dir

# Update the package lists, install make
RUN apt-get update && apt-get install -y make

# Set the working directory in the container
WORKDIR /app

# Copy only the dependency files first to leverage caching
COPY pyproject.toml poetry.lock ./

# Install only production dependencies using Poetry
RUN poetry install

# Copy the rest of the application code
COPY . .

RUN poetry install --only-root

# Download images and create vector database
RUN make create-database

# Expose the port for FastAPI
EXPOSE 8000

# Set the entrypoint to run the API directly
CMD ["poetry", "run", "fastapi", "run", "scripts/api/app.py"]
