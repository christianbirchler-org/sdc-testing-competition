# Evaluation Tool
This is the tool that performs trivial evaluation of the implemented test selection tool.

## Usage
The required test data is stored with Git LFS.
Hence you need to have Git LFS installed.
For this, refer to the offical documentation: https://git-lfs.com

After you have installed Git LFS, you can pull the large test data file(s) of this repository.
```bash
git lfs pull
```

Then build and run the evaluator tool that uses the sample dataset and performs an initial evaluation:
```bash
cd evaluator
docker build -t evaluator-image .
docker run --rm --name evaluator-container -t evaluator-image -u <toolHost:toolPort>
```

### Example
```bash
cd evaluator
docker build -t evaluator-image .
docker run --rm --name evaluator-container -t evaluator-image -u host.docker.internal:4545
```

All participants should ensure that the evaluator is able to provide an evaluation report to the console.
This is a clear indication the the gRPC interfaces are working properly, which is crucual for the competition's evaluation.
