# Evaluation Tool
This is the tool that performs trivial evaluation of the implemented test selection tool.

## Use
Build and run the evaluator tool that uses the sample dataset and performs an initial evaluation:
```bash
cd evaluator
docker build -t evaluator-image .
docker run --rm --name evaluator-container -t evaluator-image -u host.docker.internal:4545
```

All participants should ensure that the evaluator is able to provide an evaluation report to the console.
This is a clear indication the the gRPC interfaces are working properly, which is crucual for the competition's evaluation.
