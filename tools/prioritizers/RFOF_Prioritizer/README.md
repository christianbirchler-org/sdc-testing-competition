# Sample Prioritization Tool

docker build -t my-prioritizer-image .
docker run --rm --name my-prioritizer-container -t -p 4545:4545 my-prioritizer-image -p 4545

# Build the evaluator image
```bash
docker build -t evaluator-image .
```

# Run the evaluator
```bash
docker run --rm --name evaluator-container -t -v "${PWD}:/app" evaluator-image -t sample_tests/sdc-test-data.json -u host.docker.internal:4545/
```
