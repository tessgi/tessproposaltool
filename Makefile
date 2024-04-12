.PHONY: all black isort flake8 pytest

# Run all the checks which do not change files
all: isort black pytest

# Run the unit tests using `pytest`
pytest:
	poetry run pytest src tests

# Automatically format the code using `black`
black:
	poetry run black src tests

# Order the imports using `isort`
isort:
	poetry run isort src tests

# Lint the code using `flake8`
flake8:
	poetry run flake8 src tests

# Release a version 
release:
	git commit -m "Bump version to $$(poetry version -s)"
	git tag -a "v$$(poetry version -s)" -m "Release v$$(poetry version -s)"
	git push origin main --tags