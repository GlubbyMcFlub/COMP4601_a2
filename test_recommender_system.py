import unittest
import os
from recommender_system import RecommenderSystem

class TestRecommenderSystem(unittest.TestCase):
    '''
    Test suite for RecommenderSystem class
    '''

    def setUp(self):
        # Initialize RecommenderSystem with a small dataset for testing
        self.input_directory = "input"
        self.output_directory = "results"
        self.files = [f for f in os.listdir(self.input_directory) if os.path.isfile(os.path.join(self.input_directory, f))]

    def run_test_for_file(self, file_path, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, filter_type, results_file_path):
        '''
        Run a test for a given file and parameters
        '''
        
        recommender_system = RecommenderSystem(
            file_path, algorithm, output_file=results_file_path,
            neighbourhood_size=neighbourhood_size, similarity_threshold=similarity_threshold,
            include_negative_correlations=include_negative_correlations, filter_type=filter_type
        )

        result = recommender_system.run()
        
        return result
        #return self.assertIsNotNone(result, f"Test failed for file {file_path}")
        
    def test_all_files(self):
        '''
        Run tests for all files in input directory
        '''
        # Permutation
        algorithms = [RecommenderSystem.ITEM_BASED_ALGORITHM, RecommenderSystem.USER_BASED_ALGORITHM]#, RecommenderSystem.ITEM_BASED_ALGORITHM]#, RecommenderSystem.ITEM_BASED_ALGORITHM]#, RecommenderSystem.ITEM_BASED_ALGORITHM]
        parameters = [3]#, RecommenderSystem.SIMILARITY_THRESHOLD]#[RecommenderSystem.TOP_K_NEIGHBOURS, RecommenderSystem.SIMILARITY_THRESHOLD]
        neighbourhood_sizes = [2, 5, 10, 100, 700]
        similarity_thresholds = [0.9, 0.5, 0.1]
        include_negative_correlations_values = [False, True]
        results_file_path = os.path.join(self.output_directory, "results2.txt")
        file = self.files[0]
        file_path = os.path.join(self.input_directory, file)

        for algorithm in algorithms:
            for parameter in parameters:
                if parameter != RecommenderSystem.SIMILARITY_THRESHOLD:
                    for neighbourhood_size in neighbourhood_sizes:
                        for include_negative_correlations in include_negative_correlations_values:
                            if (not parameter == RecommenderSystem.TOP_K_NEIGHBOURS):
                                for similarity_threshold in similarity_thresholds:
                                    self.run_test_for_file(file_path, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, parameter, results_file_path)
                            else:
                                self.run_test_for_file(file_path, algorithm, neighbourhood_size, 0, include_negative_correlations, parameter, results_file_path)
                else:
                    for include_negative_correlations in include_negative_correlations_values:
                        for similarity_threshold in similarity_thresholds:
                            self.run_test_for_file(file_path, algorithm, 0, similarity_threshold, include_negative_correlations, parameter, results_file_path)
if __name__ == '__main__':
    unittest.main()