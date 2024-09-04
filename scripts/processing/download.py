import concurrent.futures
import csv
import time
from pathlib import Path

import requests

IMAGES_FOLDER = "data/images"


def download_image(url: str) -> None:
    """Download image from url and save it to filename"""
    filepath = Path(IMAGES_FOLDER, url.split("/")[-1])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as handle:
        response = requests.get(url, stream=True)

        # Created file is left empty if error is raised.
        response.raise_for_status()

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


if __name__ == "__main__":

    INDEX_FILEPATH = "data/index.csv"

    with open(INDEX_FILEPATH, "r") as handle:
        reader = csv.reader(handle)
        urls = [r[0] for i, r in enumerate(reader) if i > 0]

    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_image, urls)

    print(f"Downloaded {len(urls)} images in {time.perf_counter() - t:.2f} seconds")
