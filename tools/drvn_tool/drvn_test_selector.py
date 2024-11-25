import random
import argparse
import competition_pb2_grpc
import competition_pb2
import grpc
import concurrent.futures as fut
import io
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import numpy as np
import shapely
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from shapely.geometry import LineString

# Simple GA for sorting larges lists
class GeneticAlgorithmSelector:
    def __init__(self, tests, population_size=50, generations=100, subset_size=20, mutation_rate=0.1, alpha=0.5):
        """
        Initialize the genetic algorithm.

        :param tests: List of test cases with curvature profiles and road lengths.
        :param population_size: Number of individuals in each generation.
        :param generations: Number of generations to run the algorithm.
        :param subset_size: Maximum size of the subset to select.
        :param mutation_rate: Probability of mutating an individual.
        :param alpha: Weighting factor between diversity and road length in fitness function.
        """
        self.tests = tests
        self.population_size = population_size
        self.generations = generations
        self.subset_size = subset_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha

    # Calculate diversity as mean pairwise distance
    def calculate_fitness(self, subset):
        profiles = [self.tests[i]['curvature_profile'] for i in subset]
        diversity = np.mean([self.calculate_pairwise_diversity(a, b) for a in profiles for b in profiles if a is not b])
        
        return diversity

    # Compute pairwise diversity (Euclidean distance).
    def calculate_pairwise_diversity(self, profile1, profile2):
        return np.linalg.norm(np.array(profile1) - np.array(profile2))

    # Generate an initial random population.
    def initialize_population(self):
        return [random.sample(range(len(self.tests)), self.subset_size) for _ in range(self.population_size)]

    # Perform crossover between two parents.
    def crossover(self, parent1, parent2):
        point = random.randint(1, self.subset_size - 1)
        child = list(set(parent1[:point] + parent2[point:]))
        return random.sample(child, self.subset_size) if len(child) > self.subset_size else child

    # Randomly mutate an individual.
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # Randomly select an index in the individual
            idx_to_mutate = random.randint(0, len(individual) - 1)
            
            # Replace the selected index with a random index from the entire tests list
            individual[idx_to_mutate] = random.randint(0, len(self.tests) - 1)

    # Run the genetic algorithm.
    def run(self):
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness of each individual
            fitness_scores = [self.calculate_fitness(individual) for individual in population]
            
            # Select the top individuals for mating
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
            population = sorted_population[:self.population_size // 2]
            
            # Generate the next generation via crossover
            next_generation = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)
            
            # Combine parents and children for the next generation
            population += next_generation
        
        # Return the best solution
        best_individual = max(population, key=self.calculate_fitness)
        return best_individual

# DRVN Test Selector
class DRVNTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    # Sample input data
    PRED_VAL = 0.6
    GREEDY_DIVERSITY = 0.9

    # Initiate some variables, default algorithm is greedy
    def __init__(self, model, algorithm='greedy') -> None:
        """
        Initialize the DRVN class.

        :param model_location: String containing the location of the model to use.
        :param model: Tensorflow model.
        :param algorithm: The search algorithm to use.
        """
        super().__init__()
        self.model_location = model
        self.model = None
        self.algorithm = algorithm

    def Name(self, request, context):
        return competition_pb2.NameReply(name="drvn_ml_based_greedy_test_selector")

    def Initialize(self, request_iterator, context):
        # Load the model
        print(f"LOADING MODEL: {self.model_location}")
        self.model = tf.keras.models.load_model(self.model_location)
        return competition_pb2.InitializationReply(ok=True)

    def calculate_line_length(self, points):
        total_length = 0
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Euclidean distance
            total_length += distance
        return total_length
    
    def get_road_points(self, road_points):
        points_list = []
        for road_point in road_points:
            points_list.append((road_point.x, road_point.y))
        
        return points_list

    # Function to load and preprocess data for a single test case
    def preprocess_single_test(self, test_case):
        # Extract ID, image, and auxiliary data
        test_id = test_case.testId
        road_points = self.get_road_points(test_case.roadPoints)

        road_shape = LineString(road_points)

        x, y = road_shape.xy

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(2.24, 2.24))

        plt.cla()  # Clear the current axis

        ax.plot(x, y, color='black', linewidth=2, marker='o')

        # Hide axis
        ax.set_axis_off()

        road_length = self.calculate_line_length(road_points)
        # print(road_length)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save the figure to a BytesIO object
        buf.seek(0)
        img_byte_arr = buf.getvalue()
        buf.close()

        # Decode the image and reshape if needed
        image = tf.io.decode_png(img_byte_arr, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to [0, 1] range

        road_length = road_length_tensor = tf.convert_to_tensor([[self.calculate_line_length(road_points)]], dtype=tf.float32)

        test_data = {
            'id': test_id,
            'road_length': road_length,
            'road_points': road_points
        }

        return (test_data, image, road_length)
    
    # Batch predict and filter based on threshold
    def batch_predict(self, model, data):
        tests_to_keep = []
        tests_data, images, aux_data = [], [], []

        # Extract IDs, images, and auxiliary data
        for test_data, image, aux in data:
            tests_data.append(test_data)

            images.append(image)
            aux_data.append(aux)

            # Check if batch is full
            if len(images) == 32:
                # predictions = model.predict((np.stack(images), np.stack(aux_data)))
                predictions = model.predict((np.stack(images)))
                tests_to_keep.extend([dict(test_, prediction=pred) for test_, pred in zip(tests_data, predictions) if pred >= self.PRED_VAL])
                
                # Clear batch
                tests_data, images, aux_data = [], [], []

        # Process any remaining data in the final batch
        if images:
            predictions = model.predict((np.stack(images)))
            tests_to_keep.extend([dict(test_, prediction=pred) for test_, pred in zip(tests_data, predictions) if pred >= self.PRED_VAL])

        return tests_to_keep

    # Compute the diversity between two curvature profiles using Euclidean distance.
    def calculate_pairwise_diversity(self, profile1, profile2):
        return np.linalg.norm(np.array(profile1) - np.array(profile2))
        # return euclidean(profile1, profile2)

    # Compute incremental diversity as the mean distance between new profile and all selected profiles.
    def calculate_incremental_diversity(self, selected_profiles, new_profile):
        distances = [self.calculate_pairwise_diversity(profile['curvature_profile'], new_profile) for profile in selected_profiles]
        return np.mean(distances)

    def greedy_diverse_selection(self, tests, diversity_threshold=1):
        # Initialize with the first profile as a random starting point
        # start with a randomly selected indice
        indice = random.randint(0,len(tests)-1)
        selected_indices = [indice]
        selected_profiles = [tests[indice]]
        
        while True:
            max_diversity = -np.inf # Always start with a fresh max
            best_candidate = None # Best candidate selected by the greedy algorithm
            stopping_condition = len(selected_indices) # Stopping condition based on whether we increase the number of items
            
            # Check each unselected line to find the one that maximizes incremental diversity
            for i in range(len(tests)):
                if i in selected_indices:
                    continue
                
                # print(i)
                # Calculate incremental diversity for the current candidate
                incremental_diversity = self.calculate_incremental_diversity(selected_profiles, tests[i]['curvature_profile'])
                
                # We want to ensure a minumum amount of diversity
                # Set current to max if it is larger than max
                if incremental_diversity > diversity_threshold and incremental_diversity > max_diversity:
                    max_diversity = incremental_diversity
                    best_candidate = i

            # Add the best candidate to the selected subset
            if best_candidate:
                selected_indices.append(best_candidate)
                selected_profiles.append(tests[best_candidate])

            # Break loop when we reach our stopping condition
            if len(selected_indices) <= stopping_condition:
                break
        
        # Sort the combined list based on the road_length in selected_profiles
        combined_sorted = sorted(list(zip(selected_profiles, selected_indices)), key=lambda x: x[0]['road_length'])

        # Unzip the sorted list back into two separate lists
        sorted_profiles, sorted_indices = zip(*combined_sorted)

        return sorted_profiles, list(sorted_indices)

    # Taken from eval code to get the curvature profile of the road
    def _curvature_profile(self, data) -> list[float]:
        # road_points = data[0]['road_points']
        road_points = data['road_points']
        # print("compute curvature profile")
        road_shape = shapely.LineString(road_points)

        delta_s = 2  # 3 meters

        curvature_profile = np.zeros(int(road_shape.length)) # we want the curvature for every meter
        for s in range(len(curvature_profile)):
            #s = (i+1)*delta_s

            # ignore the edge cases close to the ends of the road
            if (s < delta_s/2) or (s > road_shape.length-delta_s/2):
                continue


            pt_q: shapely.Point = road_shape.interpolate(s-delta_s, normalized=False)
            pt_r: shapely.Point = road_shape.interpolate(s-delta_s/2, normalized=False)

            pt_s: shapely.Point = road_shape.interpolate(s, normalized=False)

            pt_t: shapely.Point = road_shape.interpolate(s+delta_s/2, normalized=False)
            pt_u: shapely.Point = road_shape.interpolate(s+delta_s, normalized=False)

            tangent_r_vec = np.array((pt_s.x-pt_q.x, pt_s.y-pt_q.y))
            tangent_t_vec = np.array((pt_u.x-pt_s.x, pt_u.y-pt_s.y))

            cos_phi = np.dot(tangent_r_vec, tangent_t_vec)/(np.linalg.norm(tangent_r_vec)*np.linalg.norm(tangent_t_vec))
            phi = np.arccos(cos_phi)

            kappa = phi/delta_s
            if np.isnan(kappa):
                continue

            curvature_profile[s] = kappa
        
        x = np.linspace(0, 1, len(curvature_profile))  # Original scale
        f = interp1d(x, curvature_profile, kind='linear')
        x_new = np.linspace(0, 1, 300)
        resampled_profile = f(x_new)

        data['curvature_profile'] = resampled_profile.ravel()

        return data

    def Select(self, request_iterator, context):
        # Grab all the test cases for optimization
        all_test_cases = []
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            all_test_cases.append(sdc_test_case)

        print(f"TOTAL TEST CASES: {len(all_test_cases)}")

        # Pre-process the test case data to get the model input data
        data = []
        for test in all_test_cases:
            data.append(self.preprocess_single_test(test))
        
        print("PREPARING TO PREDICT!")
        # Run the test data through our model to get an initial set of predictions
        filtered_tests = self.batch_predict(self.model, data)

        # Compute curvature profile for the selected tests
        with ThreadPoolExecutor() as executor:
            modified_tests = list(tqdm(executor.map(self._curvature_profile, filtered_tests), total=len(filtered_tests)))

        # Run the search algorithm to find the best diversity
        if self.algorithm == 'greedy':
            print(f"USING GREEDY SEARCH")
            selected_profiles, selected_indices = self.greedy_diverse_selection(modified_tests, self.GREEDY_DIVERSITY)
        else:
            print(f"USING GA SEARCH")
            ga_selector = GeneticAlgorithmSelector(modified_tests, population_size=20, generations=50, subset_size=20, mutation_rate=0.1, alpha=0.5)
            best_subset = ga_selector.run()
            selected_indices = sorted(best_subset, key=lambda i: modified_tests[i]['road_length'])

        # Print the selected indices
        print("Selected Indices:", selected_indices)

        # Return the selected tests for evaluation
        for idx in selected_indices:
            yield competition_pb2.SelectionReply(testId=modified_tests[idx]['id'])

if __name__ == "__main__":
    print("start test selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    parser.add_argument('-m', '--model', type=str, nargs="?", default='./model/matplot_img_vgg19_aug.keras', const='./model/matplot_img_vgg19_aug.keras', help="The location of the model we are using.")
    parser.add_argument('-a', '--algorithm', type=str, nargs="?", default='greedy', const='greedy', help="The serach algorithm to deploy")
    args = parser.parse_args()
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=8))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(DRVNTestSelector(model=args.model, algorithm=args.algorithm), server)

    server.add_insecure_port(GRPC_URL)
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")
