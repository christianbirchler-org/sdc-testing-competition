# Random Test Selector
## Usage
```bash
cd tools/sample_tool
docker build -t my-selector-image .
docker run --rm --name my-selector-container -t -p 4545:4545 my-selector-image -p 4545
```

