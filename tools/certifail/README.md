# rf_static_test_selector

This is a sample implementation of a test selector tool that uses random forests to predict whether a test will pass or fail before the test is actually simulated. The tool leverages a Docker image to streamline usage, eliminating the need for complicated environment setups.

The test selector relies on the prediction outcomes from a pretrained random forest model, and XGBoost and a Naive bayes model. The models takes as input a DataFrame of road features and returns a prediction. A test is selected if all 3 predict that the test will fail.

The features for a test are calculated using the `calculate_features` function in the `features.py` file. This function takes the road points defined in the test sample as input and returns a set of features, including road length, direct length, right and left turns, and other related metrics.

## Usage

To run the selector tool, use Docker. Follow the instructions below to set up the Docker image and run the Docker container:

1. Navigate to the tool directory:
   ```bash
   cd tools/sample_tool
   
2. Build the Docker image::
    ```bash
    docker build -t my-selector-image .

3. Run the Docker container:
    ```bash
    docker run --rm --name my-selector-container -t -p 4545:4545 certifail-image -p 4545
    ```

This will start the test selector tool, making it accessible through port 4545.

## PreBuilt Docker Image 

https://hub.docker.com/r/fasihmunirmalik/certifail


docker pull fasihmunirmalik/certifail:latest