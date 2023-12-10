import unittest
import os
import concurrent.futures
from recommender_system import RecommenderSystem

class TestRecommenderSystem(unittest.TestCase):
    '''
    Test suite for RecommenderSystem class
    
    This test suite is designed to test the RecommenderSystem class permutations to easily find the best parameters for the algorithms
    '''

    def setUp(self):
        '''
        Set up the test suite directories and files
        '''
        self.input_directory = "input"
        self.output_directory = "results"
        self.files = [f for f in os.listdir(self.input_directory) if os.path.isfile(os.path.join(self.input_directory, f))]
    
    def generate_results_file_name(self, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, filter_type):
        '''
        Generate a dynamic results file name based on permutations
        '''

        algorithm_str = "item" if algorithm == RecommenderSystem.ITEM_BASED_ALGORITHM else "user"
        filter_type_str = ""
        
        if filter_type == RecommenderSystem.TOP_K_NEIGHBOURS:
            filter_type_str = "topk"
        elif filter_type == RecommenderSystem.SIMILARITY_THRESHOLD:
            filter_type_str = "similarity_threshold"
        else:
            filter_type_str = "hybrid"

        file_name = f"results_{algorithm_str}_{filter_type_str}_ns{neighbourhood_size}_sim{similarity_threshold}_{'negcorr_true' if include_negative_correlations else 'negcorr_false'}.txt"
        return os.path.join(self.output_directory, file_name)

    def run_test_for_file(self, file_path, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, filter_type, results_file_path):
        '''
        Run a test for a given file and permutation
        Returns the result (MAE) of the test
        '''

        recommender_system = RecommenderSystem(
            file_path, algorithm, output_file=results_file_path,
            neighbourhood_size=neighbourhood_size, similarity_threshold=similarity_threshold,
            include_negative_correlations=include_negative_correlations, filter_type=filter_type
        )

        result = recommender_system.run()

        return result

    def test_all_files(self):
        '''
        Run tests for all files in input directory (currently hardcoded to run only for the first file -> assignment2 data)
        '''
        # Permutation parameters
        algorithms = [RecommenderSystem.USER_BASED_ALGORITHM, RecommenderSystem.ITEM_BASED_ALGORITHM]
        filter_types = [RecommenderSystem.SIMILARITY_THRESHOLD, 3, RecommenderSystem.TOP_K_NEIGHBOURS]
        neighbourhood_sizes = [2, 5, 10, 100, 700]
        similarity_thresholds = [0.9, 0.5, 0.1]
        include_negative_correlations_values = [False, True]
        
        # Data file
        file = self.files[0]
        file_path = os.path.join(self.input_directory, file)

        # Multithreaded execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            # Run tests for all permutations
            for algorithm in algorithms:
                for filter_type in filter_types:
                    if filter_type != RecommenderSystem.SIMILARITY_THRESHOLD:
                        # Use neighbourhood size only for top-k and hybrid filter_type
                        for neighbourhood_size in neighbourhood_sizes:
                            for include_negative_correlations in include_negative_correlations_values:
                                if filter_type != RecommenderSystem.TOP_K_NEIGHBOURS:
                                    # Use similarity threshold only for hybrid filter_type
                                    for similarity_threshold in similarity_thresholds:
                                        results_file_path = self.generate_results_file_name(
                                            algorithm, neighbourhood_size, similarity_threshold,
                                            include_negative_correlations, filter_type
                                        )
                                        future = executor.submit(self.run_test_for_file, file_path, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, filter_type, results_file_path)
                                        futures.append(future)
                                else:
                                    # Ignore similarity threshold if top-k filter_type
                                    similarity_threshold = 0
                                    results_file_path = self.generate_results_file_name(
                                        algorithm, neighbourhood_size, similarity_threshold,
                                        include_negative_correlations, filter_type
                                    )
                                    future = executor.submit(self.run_test_for_file, file_path, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, filter_type, results_file_path)
                                    futures.append(future)
                    else:
                        # Ignore neighbourhood size if similarity threshold
                        for include_negative_correlations in include_negative_correlations_values:
                            for similarity_threshold in similarity_thresholds:
                                neighbourhood_size = 0
                                results_file_path = self.generate_results_file_name(
                                    algorithm, neighbourhood_size, similarity_threshold,
                                    include_negative_correlations, filter_type
                                )
                                future = executor.submit(self.run_test_for_file, file_path, algorithm, neighbourhood_size, similarity_threshold, include_negative_correlations, filter_type, results_file_path)
                                futures.append(future)

            # Wait for all threads to complete
            concurrent.futures.wait(futures)

if __name__ == '__main__':
    unittest.main()
