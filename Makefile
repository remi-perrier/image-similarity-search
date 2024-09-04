.PHONY: clear-images
clear-images:
	rm -rf data/images

.PHONY: download-images
download-images: clear-images
	mkdir -p data/images
	poetry run python scripts/processing/download.py

.PHONY: create-database
create-database: download-images
	poetry run python scripts/processing/create_database.py

.PHONY: run-api-prod
run-api-prod: create-database	
	poetry run fastapi run scripts/api/app.py