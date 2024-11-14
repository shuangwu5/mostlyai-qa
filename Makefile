.PHONY: help
help: ## show definition of each function
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z1-9_-]+:.*?## / {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: clean
clean: ## remove .gitignore files
	git clean -fdX

.PHONY: install
install: ## install dependencies
	poetry install

.PHONY: lint
lint: ## run lints
	poetry run pre-commit run --all-files

.PHONY: test
test: ## run tests
	poetry run pytest

.PHONY: all
all: clean install lint test ## run all commands

.PHONY: build
build: ## build package
	poetry build
	twine check --strict dist/*

.PHONY: examples
examples: ## run all examples
	find ./examples -maxdepth 1 -type f -name "*.ipynb" -print -execdir jupyter nbconvert --to script {} \;
	find ./examples -maxdepth 1 -type f -name "*.py" -print -execdir python {} \;
