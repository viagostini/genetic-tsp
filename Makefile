MAKE := make --no-print-directory

build:
	@docker build -t genetic-tsp .

run:
	@docker run --rm -p 8501:8501 genetic-tsp

dev:
	@docker run --rm -p 8501:8501 -v $(shell pwd):/usr/src/app genetic-tsp