# DRVN- Self-Driving car test selection tool

DRVN is a tool for creating self-driving car regression suites. It uses a fine-tuned CNN to infer if a test-case could fail or not, based on the road image given to it.

It then uses a greedy algorithm to search through the selected roads, selecting those with the best diversity.

## Building the tool
DRVN can and should be run from docker and a Dockerfile has been provided for this.

```
cd tools/drvn_tool
docker build --tag drvn_tool .
```

This builds the drvn_tool and creates a docker image `drvn_tool`.

## Running the tool
The default model has been built into the Dockerfile so it is simple to run.
```
docker run --rm --name drvn_selector --gpus all -t -p 4545:4545 drvn_tool -p 4545
```
This will start the selector, run this with `-d` if you want the terminal free still and then check the container with `docker logs -f drvn_selector`.

When the GRPC server shows as running, you can then run the evaluator against the selector.

## Running other configurations
The default configuration has been set for the competition. However, we have configured the tool to use both a greedy and GA diversity selection algorithm. These can be defined at run-time, along with other models.

When using other models, we recommend creating a volume of the model folder, to be able to use these at will.
```
docker run --rm --name drvn_selector --gpus all -t -p 4545:4545 -v $(pwd)/model:/app/model:rw drvn_tool:tf2.18 -p 4545 -m ./model/matplot_img_vgg16.keras -a 'ga'
```

This will run the vgg16 model and use the GA algorithm.