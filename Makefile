DATA_DIR = ~/.fdua-competition

download:
	mkdir -p ${DATA_DIR}/downloads
	gsutil -m cp -r gs://yo-personal/fdua/downloads/* ${DATA_DIR}/downloads


upload-secret:
	gsutil cp .env gs://yo-personal/fdua/secret/

download-secret:
	gsutil cp gs://yo-personal/fdua/secret/.env .

upload: clean-data-dir
	gsutil -m cp -r ${DATA_DIR}/downloads gs://yo-personal/fdua/

unzip:
	cd ${DATA_DIR}/downloads && \
	find . -name "*.zip" -print -exec unzip {} \;
	make clean-data-dir

clean-project:
	-rm -rf .venv
	-rm uv.lock
	find . -type f -name ".DS_Store" -print -exec rm -r {} +
	find . -type d -name "__pycache__" -print -exec rm -r {} +
	find . -type d -name ".pytest_cache" -print -exec rm -r {} +
	find . -type d -name ".ruff_cache" -print -exec rm -r {} +
	find . -type f -name "*.Identifier" -print -exec rm -r {} +

clean-data-dir:
	find ${DATA_DIR} -type d -name "*__MACOS*" -print -exec rm -r {} +
	find ${DATA_DIR} -type f -name ".DS_Store" -print -exec rm -r {} +
	find ${DATA_DIR} -type d -name "__pycache__" -print -exec rm -r {} +
	find ${DATA_DIR} -type f -name "*.Identifier" -print -exec rm -r {} +

lint:
	uv run isort .
	uv run ruff check . --fix
	uv run ruff format .

pre-commit: lint clean-project

test:
	cp tests/assets/1.pdf ${DATA_DIR}/downloads/documents/
	uv run pytest -vvv

backup-repo: clean-project
	gsutil -m rm -r gs://yo-personal/fdua/repo/fdua-competition
	gsutil -m cp -r . gs://yo-personal/fdua/repo/fdua-competition

run:
	uv run python -m fdua_competition.main