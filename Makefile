.PHONY: build upload clean

build:
	rm -rf dist traceroot.egg-info/
	uv build

upload:
	uvx twine check dist/*
	uvx twine upload dist/*

clean:
	rm -rf dist traceroot.egg-info/
