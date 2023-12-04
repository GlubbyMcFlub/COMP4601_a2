import unittest
import os
import datetime
from predict import RecommenderSystem

class TestRecommenderSystem(unittest.TestCase):
    '''
    Test suite for RecommenderSystem class
    '''

    def setUp(self):
        # Initialize RecommenderSystem with a small dataset for testing
        self.input_directory = "input"
        self.output_directory = "results"
        self.files = [f for f in os.listdir(self.input_directory) if os.path.isfile(os.path.join(self.input_directory, f))]

    def run_test_for_file(self, file_path, algorithm, parameter, neighbourhood_size, similarity_threshold, include_negative_correlations):
        '''
        Run a test for a given file and parameters
        '''
        recommender_system = RecommenderSystem(
            file_path, algorithm, parameter,
            neighbourhood_size=neighbourhood_size, similarity_threshold=similarity_threshold,
            include_negative_correlations=include_negative_correlations
        )

        self.assertIsNotNone(recommender_system.run())

    def test_all_files(self):
        '''
        Run tests for all files in input directory
        '''
        # Permutation
        algorithms = [RecommenderSystem.ITEM_BASED_ALGORITHM, RecommenderSystem.USER_BASED_ALGORITHM]
        parameters = [RecommenderSystem.TOP_K_NEIGHBOURS, RecommenderSystem.SIMILARITY_THRESHOLD]
        neighbourhood_sizes = [2, 5, 10]
        similarity_thresholds = [0.2, 0.5, 0.8]
        include_negative_correlations_values = [True, False]

        run_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        results_file_path = os.path.join(self.output_directory, f"test_results_{run_timestamp}.txt")

        with open(results_file_path, "w", encoding=None) as results_file:
            for file in self.files:
                file_path = os.path.join(self.input_directory, file)
                with self.subTest(file=file):
                    for algorithm in algorithms:
                        for parameter in parameters:
                            for neighbourhood_size in neighbourhood_sizes:
                                for similarity_threshold in similarity_thresholds:
                                    for include_negative_correlations in include_negative_correlations_values:
                                        results_file.write(f"File: {file}\n")
                                        results_file.write(f"Algorithm: {algorithm}, Parameter: {parameter}, Neighbourhood Size: {neighbourhood_size}, Similarity Threshold: {similarity_threshold}, Include Negative Correlations: {include_negative_correlations}\n")
                                        self.run_test_for_file(file_path, algorithm, parameter, neighbourhood_size, similarity_threshold, include_negative_correlations)
                                        results_file.write(f"Test Result: Passed\n\n")


if __name__ == '__main__':
    unittest.main()