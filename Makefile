.PHONY: help install demo serve test osm web clean

help:
	@echo "make install   install dependencies"
	@echo "make demo      run the full pipeline, write artifacts + demo data"
	@echo "make serve     run the FastAPI dispatcher service on :8000"
	@echo "make web       serve the static demo on :8080"
	@echo "make test      run the test suite"
	@echo "make osm       fetch a real OpenStreetMap graph for Bengaluru"

install:
	pip install -r requirements.txt

demo:
	python -m routeshield.cli demo

serve:
	python -m routeshield.cli serve

web:
	cd web && python -m http.server 8080

test:
	python -m pytest -q

osm:
	python scripts/fetch_osm.py --city bengaluru --out artifacts/osm_ways.json
	@echo "Now run:  ROUTESHIELD_OSM=artifacts/osm_ways.json make demo"

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache artifacts/*.db
