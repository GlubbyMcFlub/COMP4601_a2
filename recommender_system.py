import os
import math
import numpy as np
import time
import logging
import threading
class RecommenderSystem:
    '''
    RecommenderSystem class for COMP4601 Assignment 2.
    This class implements the item-based and user-based collaborative filtering algorithms 
        and provides methods for running the algorithms and finding the Mean Absolute Error (MAE).
    '''
    # Algorithm/filter constants
    ITEM_BASED_ALGORITHM = 0
    USER_BASED_ALGORITHM = 1
    TOP_K_NEIGHBOURS = 0
    SIMILARITY_THRESHOLD = 1
    # Default values
    MISSING_RATING = 0.0
    DEFAULT_NEIGHBOURHOOD_SIZE = 2
    DEFAULT_SIMILARITY_THRESHOLD = 0.5
    CLOSE_TO_ZERO_TOLERANCE = 1e-5
    # Rating thresholds for clipping the final predicted rating
    MIN_RATING = 0.5
    MAX_RATING = 5.0

    def __init__(self, path, algorithm, output_file, neighbourhood_size=DEFAULT_NEIGHBOURHOOD_SIZE, close_to_zero_tolerance=CLOSE_TO_ZERO_TOLERANCE, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, include_negative_correlations=False, filter_type=TOP_K_NEIGHBOURS):
        """
        Initialize the RecommenderSystem object.
        Parameters:
        - path (str): Path to the input file.
        - algorithm (int): Algorithm to use for the recommender system.
        - output_file (str): Path to the output file.
        - neighbourhood_size (int): Size of the neighbourhood to use for prediction.
        - close_to_zero_tolerance (float): Tolerance for similarity values close to zero.
        - similarity_threshold (float): Threshold for similarity values.
        - include_negative_correlations (bool): Whether to include negative correlations in the similarity calculations.
        - filter_type (int): Type of filter to use for the recommender system.
        Returns:
        None
        """
        # Initialize variables
        self.logger_lock = threading.Lock()
        self.path = path
        self.algorithm = algorithm
        self.output_file = output_file
        self.logger = self.configure_logger()
        self.close_to_zero_tolerance = close_to_zero_tolerance
        self.neighbourhood_size = neighbourhood_size
        self.similarity_threshold = similarity_threshold
        self.include_negative_correlations = include_negative_correlations
        self.filter_type = filter_type
        self.num_users, self.num_items, self.users, self.items, self.ratings = self.read_data()
        
        # For data analysis
        self.min_filtered_neighbourhood_size = float('inf')
        self.max_filtered_neighbourhood_size = 0
        self.min_neighbours_size = float('inf')
        self.max_neighbours_size = 0
        
    def configure_logger(self):
        '''
        Configure the logger for the RecommenderSystem instance.
        Parameters: None
        Returns: logger (logging.Logger): Logger for the RecommenderSystem instance.
        '''
        with self.logger_lock:
            # Create a new logger instance for each RecommenderSystem instance
            recommender_logger = logging.getLogger(f"{__name__}_{id(self)}")
            recommender_logger.propagate = False  # Disable propagation to the root logger

            # Remove existing handlers to prevent duplicates
            for handler in recommender_logger.handlers[:]:
                recommender_logger.removeHandler(handler)

            formatter = logging.Formatter("%(message)s")

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(self.output_file)
            file_handler.setFormatter(formatter)

            recommender_logger.setLevel(logging.INFO)
            recommender_logger.addHandler(stream_handler)
            recommender_logger.addHandler(file_handler)

            return recommender_logger
        
    def close_logger(self):
        '''
        Close the logger for the RecommenderSystem instance.
        Parameters: None
        Returns: None
        '''
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                if handler.stream is not None:
                    handler.close()
                self.logger.removeHandler(handler)

    def read_data(self):
        """
        Read the data from the input file to extract the number of users, number of items, users, items, and ratings.
        Parameters: None
        Returns:
        Tuple containing:
        - num_users (int): Number of users.
        - num_items (int): Number of items.
        - users (numpy.ndarray): Array of users.
        - items (numpy.ndarray): Array of items.
        - ratings (numpy.ndarray): Array of ratings.
        """
        try:
            with open(self.path, 'r') as file:
                num_users, num_items = map(int, file.readline().split())
                users = np.array(file.readline().split())
                items = np.array(file.readline().split())
                ratings = np.array([[float(rating) if rating != self.MISSING_RATING else self.MISSING_RATING for rating in line.split()] for line in file])
                return num_users, num_items, users, items, ratings
        except ValueError as err:
            raise ValueError("Error: %s", str(err)) from err
        except FileNotFoundError as err:
            self.logger.error("Error in read_data: %s", str(err))
            return None, None, None, None, None
    
    ###################################
    #####        Item-based       #####
    ##### Collaborative Filtering #####
    ###################################
    
    def compute_item_similarity(self, item1_index, item2_index, common_users, average_ratings):
        """
        Computes the Adjusted Cosine Similarity between two items based on their common user ratings.
        Parameters:
        - item1_index (int): Index of item 1 (item in question).
        - item2_index (int): Index of item 2.
        - common_users (list): List of indices for common users.
        - average_ratings (numpy.ndarray): Array of average ratings for each user.
        Returns:
        - similarity (float): Adjusted Cosine Similarity between the two items.
        """
        
        num_common_users = len(common_users)

        if num_common_users == 0:
            return 0
        
        common_user_item1 = []
        common_user_item2 = []
        for i in common_users:
            common_user_item1.append(self.ratings[i, item1_index] - average_ratings[i])
            common_user_item2.append(self.ratings[i, item2_index] - average_ratings[i])
        numerator = sum(x * y for x, y in zip(common_user_item1, common_user_item2))
        item1_denominator = 0
        for num in common_user_item1:
            item1_denominator += num ** 2
        item2_denominator = 0
        for num in common_user_item2:
            item2_denominator += num ** 2
        
        denominator = math.sqrt(item1_denominator * item2_denominator)
        similarity = max(0, numerator / denominator) if denominator != 0 else 0

        return similarity
    
    def precompute_item_similarities(self, user_index, item_index, average_ratings):
        """
        Precomputes the similarities between all pairs of items.
        Parameters:
        - user_index (int): Index of the user.
        - item_index (int): Index of the item.
        - average_ratings (numpy.ndarray): Array of average ratings for each user.
        Returns:
        - similarities (numpy.array): Matrix of precomputed similarities.
        """
        similarity_matrix_file = f"similarities/item_{user_index}_{item_index}_similarity_matrix.npy"

        # If the similarity matrix has already been computed, load it from the file
        if os.path.exists(similarity_matrix_file):
            similarities = np.load(similarity_matrix_file)
        else:
            similarities = np.zeros((self.num_items, self.num_items), dtype=np.float64)

            item1_ratings = np.where(self.ratings[:, item_index] != self.MISSING_RATING)[0]
            
            for j in range(self.num_items):
                if j != item_index:
                    # Find common users who rated both items item_index and j
                    item2_ratings = np.where(self.ratings[:, j] != self.MISSING_RATING)[0]
                    common_users = np.intersect1d(item1_ratings, item2_ratings)

                    # Calculate similarity between items item_index and j based on common users
                    similarity = self.compute_item_similarity(item_index, j, common_users, average_ratings)
                    similarities[item_index, j] = similarity
                    similarities[j, item_index] = similarity
                    
            os.makedirs(os.path.dirname(similarity_matrix_file), exist_ok=True)
            np.save(similarity_matrix_file, similarities)

        return similarities
    
    def predict_item_rating(self, user_index, item_index, similarities):
        """
        Predict a single rating for a user-item pair using the item-based algorithm.
        Parameters:
        - user_index (int): Index of the user.
        - item_index (int): Index of the item.
        - similarities (numpy.ndarray): 2D array containing precomputed item similarities.
        Returns:
        Tuple containing:
        - predict_rating (float): Predicted rating for the user-item pair.
        - total_similarity (float): Total similarity score of the neighbourhood.
        - adjusted_neighbourhood_size (int): Adjusted size of the neighbourhood used for prediction.
        """
        
        try:
            # Get all neighbours who rated the item, adjust neighbourhood size if necessary
            neighbours = np.where(self.ratings[user_index] != self.MISSING_RATING)[0]
            
            self.min_neighbours_size = min(self.min_neighbours_size, len(neighbours))
            self.max_neighbours_size = max(self.max_neighbours_size, len(neighbours))
            
            # Create array of tuples using neighbours indices and similarities
            if (self.include_negative_correlations):
                filtered_indices = [(num, i, i in neighbours) for i, num in enumerate(abs(similarities[item_index])) if num != 0]
            else:
                filtered_indices = [(num, i, i in neighbours) for i, num in enumerate(similarities[item_index]) if num > 0]

            # Sort array based on similarities in descending order for neighbourhood similarities
            sorted_indices = np.array([index for _, index, is_in_array in sorted(filtered_indices, key=lambda x: x[0], reverse=True) if is_in_array])

            # Filter neighbours based on filter type
            if self.filter_type == self.SIMILARITY_THRESHOLD:
                filtered_indices = [index for index in sorted_indices if similarities[item_index][index] > self.similarity_threshold]
                adjusted_neighbourhood_size = len(filtered_indices)
            elif self.filter_type == self.TOP_K_NEIGHBOURS:
                adjusted_neighbourhood_size = min(self.neighbourhood_size, len(filtered_indices))
                filtered_indices = sorted_indices[:adjusted_neighbourhood_size]
                adjusted_neighbourhood_size = len(filtered_indices)
            else:
                adjusted_neighbourhood_size = min(self.neighbourhood_size, len(filtered_indices))
                filtered_indices = [index for index in sorted_indices if similarities[item_index][index] > self.similarity_threshold][:adjusted_neighbourhood_size]
                adjusted_neighbourhood_size = len(filtered_indices)

            self.min_filtered_neighbourhood_size = min(self.min_filtered_neighbourhood_size, adjusted_neighbourhood_size)
            self.max_filtered_neighbourhood_size = max(self.max_filtered_neighbourhood_size, adjusted_neighbourhood_size)

            # If no neighbours found, use average rating for item
            if adjusted_neighbourhood_size == 0:
                ratings_without_zeros = np.where(self.ratings[user_index] != 0, self.ratings[user_index], np.nan)
                predict_rating = np.nanmean(ratings_without_zeros)
                total_similarity = 0
            else:
                # Calculate the predicted rating for the user-item pair
                sum_ratings = np.sum(similarities[item_index, filtered_indices] * self.ratings[user_index, filtered_indices])
                
                if (self.include_negative_correlations):
                    total_similarity = np.sum(abs(similarities[item_index, filtered_indices]))
                else:
                    total_similarity = np.sum(similarities[item_index, filtered_indices])
                
                if abs(total_similarity) < self.close_to_zero_tolerance:
                    ratings_without_zeros = np.where(self.ratings[user_index] != 0, self.ratings[user_index], np.nan)
                    predict_rating = np.nanmean(ratings_without_zeros)
                    total_similarity = 0
                else:
                    predict_rating = sum_ratings / total_similarity
                
                # Clip the final predicted value to be within the rating range
                predict_rating = max(self.MIN_RATING, min(predict_rating, self.MAX_RATING))

            return predict_rating, total_similarity, adjusted_neighbourhood_size
        except Exception as err:
            self.logger.error("Error in predict_rating: %s", str(err))
            return None, None, None
    
    def find_item_mae(self):
        """
        Find the Mean Absolute Error (MAE) for the item-based prediction model.
        Parameters: None
        Returns:
        - mae (float): Mean Absolute Error (MAE) for the item-based prediction model.
        """
        try:
            test_set_size = 0
            numerator = 0

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0
            # Calculate average ratings for all users, ignoring any missing ratings
            average_ratings = np.nanmean(np.where(self.ratings != self.MISSING_RATING, self.ratings, np.nan), axis=1)
            
            # Times
            start_time = time.time()
            avg_run_time = 0
            sim_run_time = 0
            pred_run_time = 0

            # Iterate through all user-item pairs
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        test_set_size += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING
                        
                        # Calculate average ratings for all users, ignoring any missing ratings
                        avg_times = time.time()
                        average_ratings[i] = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])
                        avg_run_time += time.time() - avg_times

                        # Precompute the similarities between all pairs of items
                        similarity_start_time = time.time()
                        similarities = self.precompute_item_similarities(i, j, average_ratings)
                        sim_run_time += time.time() - similarity_start_time

                        # Predict the rating for each user-item pair
                        prediction_start_time = time.time()
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_item_rating(i, j, similarities)
                        pred_run_time += time.time() - prediction_start_time

                        # Calculate components for MAE
                        if not np.isnan(predicted_rating):
                            error = abs(predicted_rating - temp)
                            numerator += error
                            
                            if error < self.MIN_RATING:
                                under_predictions += 1
                            elif error > self.MAX_RATING:
                                over_predictions += 1

                        if np.isnan(predicted_rating) or np.isinf(predicted_rating) or total_similarity == 0:
                            no_valid_neighbours += 1
                        else:
                            total_neighbours_used += adjusted_neighbourhood_size

                        self.ratings[i, j] = temp
                        average_ratings[i] = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])

            mae = numerator / test_set_size
            
            # Log results
            self.logger.info("Total predictions: %d", test_set_size)
            self.logger.info("Total under predictions (< %d): %d", 1, under_predictions)
            self.logger.info("Total over predictions (> %d): %d", 5, over_predictions)
            self.logger.info("Number of cases with no valid neighbours: %d", no_valid_neighbours)
            self.logger.info("Average neighbours used: %f", total_neighbours_used / test_set_size)
            self.logger.info("MAE: %f", mae)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                minutes, seconds = divmod(elapsed_time, 60)
                self.logger.info("Start: %d:%.3f (m:ss.mmm)", int(minutes), seconds)
            else:
                self.logger.info("Start: %.3fs", elapsed_time)
            self.logger.info("Time to do averages: %.3fs", avg_run_time)
            self.logger.info("Time to do similarities: %.3fs", sim_run_time)
            self.logger.info("Time to do predictions: %.3fs", pred_run_time)

            return mae
        except Exception as err:
            self.logger.error("Error in find_item_mae: %s", str(err))
            return None
        
    ###################################
    #####        User-based       #####
    ##### Collaborative Filtering #####
    ###################################
    
    def compute_user_similarity(self, user1_ratings, user2_ratings, common_items, mean_user1, mean_user2):
        """
        Computes the Pearson correlation coefficient (PCC) between two users based on their common item ratings.
        Parameters:
        - user1_ratings (numpy.ndarray): Array of ratings for user 1.
        - user2_ratings (numpy.ndarray): Array of ratings for user 2.
        - common_items (list): List of indices for common items.
        - mean_user1 (float): Average rating for user 1.
        - mean_user2 (float): Average rating for user 2.
        Returns:
        - correlation (float): Pearson correlation coefficient.
        """
        num_common_items = len(common_items)

        if num_common_items == 0:
            return 0

        user1_ratings_common = [user1_ratings[i] for i in common_items]
        user2_ratings_common = [user2_ratings[i] for i in common_items]

        numerator = sum((user1_ratings_common[i] - mean_user1) * (user2_ratings_common[i] - mean_user2) for i in range(num_common_items))
        denominator_user1 = math.sqrt(sum((user1_ratings_common[i] - mean_user1) ** 2 for i in range(num_common_items)))
        denominator_user2 = math.sqrt(sum((user2_ratings_common[i] - mean_user2) ** 2 for i in range(num_common_items)))

        if denominator_user1 * denominator_user2 == 0:
            return 0

        correlation = numerator / (denominator_user1 * denominator_user2)

        return correlation
    
    def precompute_user_similarities(self, user_index, item_index, average_ratings):
        """
        Precmoputes the similarities between all pairs of users.
        Parameters:
        - user_index (int): Index of the user.
        - item_index (int): Index of the item.
        - average_ratings (numpy.ndarray): Array of average ratings for each user.
        Returns:
        - similarities (numpy.array): Matrix of precomputed similarities.
        """
        try:
            similarity_matrix_file = f"similarities/user_{user_index}_{item_index}_similarity_matrix.npy"

            # If the similarity matrix has already been computed, load it from the file
            if os.path.exists(similarity_matrix_file):
                similarities = np.load(similarity_matrix_file)
            else:
                similarities = np.zeros((self.num_users, self.num_users), dtype=np.float64)

                for i in range(self.num_users):
                    if i != user_index:
                        user1_indices = np.where(self.ratings[user_index] != self.MISSING_RATING)[0]
                        user2_indices = np.where(self.ratings[i] != self.MISSING_RATING)[0]

                        # Find common items rated by both users
                        intersecting_indices = np.intersect1d(user1_indices, user2_indices)
                        common_items = intersecting_indices 

                        # Calculate similarity between users based on common items
                        similarity = self.compute_user_similarity(self.ratings[user_index], self.ratings[i], common_items, average_ratings[user_index], average_ratings[i])
                        similarities[user_index, i] = similarity
                        similarities[i, user_index] = similarity
                os.makedirs(os.path.dirname(similarity_matrix_file), exist_ok=True)
                np.save(similarity_matrix_file, similarities)

            return similarities
        except Exception as err:
            self.logger.error("Error in precompute_user_similarities: %s", str(err))
            return None
    
    def predict_user_rating(self, user_index, item_index, similarities, average_ratings):
        """
        Predict a single rating for a user-item pair using the user-based algorithm.
        Parameters:
        - user_index (int): Index of the user.
        - item_index (int): Index of the item.
        - similarities (numpy.ndarray): 2D array containing precomputed user similarities.
        - average_ratings (numpy.ndarray): Array of average ratings for each user.
        Returns:
        Tuple containing:
        - predict_rating (float): Predicted rating for the user-item pair.
        - total_similarity (float): Total similarity score of the neighbourhood.
        - adjusted_neighbourhood_size (int): Adjusted size of the neighbourhood used for prediction.w
        """
        try:
            # Get all neighbors who rated the item, adjust neighborhood size if necessary
            neighbours = np.where(self.ratings[:, item_index] != self.MISSING_RATING)[0]
            self.min_neighbours_size = min(self.min_neighbours_size, len(neighbours))
            self.max_neighbours_size = max(self.max_neighbours_size, len(neighbours))
            
            # Create array of tuples using neighbors indices and similarities
            if self.include_negative_correlations:
                filtered_indices = [(num, i, i in neighbours) for i, num in enumerate(abs(similarities[user_index])) if num != 0]
            else:
                filtered_indices = [(num, i, i in neighbours) for i, num in enumerate(similarities[user_index]) if num > 0]
                
            # Sort array based on similarities in descending order for neighborhood similarities
            sorted_indices = np.array([index for _, index, is_in_array in sorted(filtered_indices, key=lambda x: x[0], reverse=True) if is_in_array])
            
            # Filter neighbors based on filter type
            if self.filter_type == self.SIMILARITY_THRESHOLD:
                filtered_indices = [index for index in sorted_indices if similarities[user_index][index] > self.similarity_threshold]
                adjusted_neighbourhood_size = len(filtered_indices)
            elif self.filter_type == self.TOP_K_NEIGHBOURS:
                adjusted_neighbourhood_size = min(self.neighbourhood_size, len(sorted_indices))
                filtered_indices = sorted_indices[:adjusted_neighbourhood_size]
                adjusted_neighbourhood_size = len(filtered_indices)
            else:
                adjusted_neighbourhood_size = min(self.neighbourhood_size, len(sorted_indices))
                filtered_indices = [index for index in sorted_indices if similarities[user_index][index] > self.similarity_threshold][:adjusted_neighbourhood_size]
                adjusted_neighbourhood_size = len(filtered_indices)
            
            self.min_filtered_neighbourhood_size = min(self.min_filtered_neighbourhood_size, adjusted_neighbourhood_size)
            self.max_filtered_neighbourhood_size = max(self.max_filtered_neighbourhood_size, adjusted_neighbourhood_size)

            # If no neighbors found, use the average rating for the user
            if adjusted_neighbourhood_size == 0:
                ratings_without_zeros = np.where(self.ratings[user_index] != 0, self.ratings[user_index], np.nan)
                predict_rating = np.nanmean(ratings_without_zeros)
                total_similarity = 0
            else:
                numerator = 0
                denominator = 0

                for idx in filtered_indices:
                    similarity = similarities[user_index, idx]
                    rating_diff = self.ratings[idx, item_index] - average_ratings[idx]
                    numerator += similarity * rating_diff
                    denominator += abs(similarity)
                    
                if abs(denominator) < self.close_to_zero_tolerance:
                    # Handle the case where the denominator is zero to avoid division by zero
                    predict_rating = average_ratings[user_index]
                else:
                    predict_rating = average_ratings[user_index] + numerator / denominator
                    
                # Clip the final predicted value to be within the rating range
                predict_rating = max(self.MIN_RATING, min(predict_rating, self.MAX_RATING))
                total_similarity = np.sum(abs(similarities[user_index, filtered_indices]))

            return predict_rating, total_similarity, adjusted_neighbourhood_size
        except Exception as err:
            self.logger.error("Error in predict_user_rating: %s", str(err))
            return None, None, None
    
    def find_user_mae(self):
        """
        Find the Mean Absolute Error (MAE) for the user-based prediction model.
        Parameters: None
        Returns:
        - mae (float): Mean Absolute Error (MAE) for the user-based prediction model.
        """
        
        try:
            test_set_size = 0
            numerator = 0

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0

            # Calculate average ratings for all users, ignoring any missing ratings
            average_ratings = np.nanmean(np.where(self.ratings != self.MISSING_RATING, self.ratings, np.nan), axis=1)
            
            # Times
            start_time = time.time()
            avg_run_time = 0
            sim_run_time = 0
            pred_run_time = 0
            
            # Iterate through all user-item pairs
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        test_set_size += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING
                        
                        # Calculate average ratings for all users, ignoring any missing ratings
                        avg_times = time.time()
                        average_ratings[i] = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])
                        avg_run_time += time.time() - avg_times
                        
                        # Precompute the similarities between all pairs of users
                        similarities_time = time.time()
                        similarities = self.precompute_user_similarities(i, j, average_ratings)
                        sim_run_time += time.time() - similarities_time
                        
                        # Predict the rating for each user-item pair
                        predicted_times = time.time()
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_user_rating(i, j, similarities, average_ratings)
                        pred_run_time += time.time() - predicted_times
                        
                        
                        # Calculate components for MAE
                        if not np.isnan(predicted_rating):
                            error = abs(predicted_rating - temp)
                            numerator += error
                            
                            if error < self.MIN_RATING:
                                under_predictions += 1
                            elif error > self.MAX_RATING:
                                over_predictions += 1

                        if np.isnan(predicted_rating) or np.isinf(predicted_rating) or total_similarity == 0:
                            no_valid_neighbours += 1
                        else:
                            total_neighbours_used += adjusted_neighbourhood_size

                        self.ratings[i, j] = temp
                        average_ratings[i] = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])

            mae = numerator / test_set_size
            
            # Log results
            self.logger.info("Total predictions: %d", test_set_size)
            self.logger.info("Total under predictions (< %d): %d", 1, under_predictions)
            self.logger.info("Total over predictions (> %d): %d", 5, over_predictions)
            self.logger.info("Number of cases with no valid neighbours: %d", no_valid_neighbours)
            self.logger.info("Average neighbours used: %f", total_neighbours_used / test_set_size)
            self.logger.info("MAE: %f", mae)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                minutes, seconds = divmod(elapsed_time, 60)
                self.logger.info("Elapsed time: %d:%.3f (m:ss.mmm)", int(minutes), seconds)
            else:
                self.logger.info("Elapsed time: %.3fs", elapsed_time)
            self.logger.info("Time to do averages: %.3fs", avg_run_time)
            self.logger.info("Time to do similarities: %.3fs", sim_run_time)
            self.logger.info("Time to do predictions: %.3fs", pred_run_time)

            return mae
        except Exception as err:
            self.logger.error("Error in find_user_mae: %s", str(err))
            return None
        
    def run(self):
        '''
        Run the recommender system.
        Parameters: None
        Returns:
        - result (float): Result of the recommender system (MAE).
        '''
        try:
            with open(self.output_file, "a") as results_file:
                with self.logger_lock:
                    if (self.filter_type == self.TOP_K_NEIGHBOURS):
                        filter_type = "Top-K Neighbours"
                    elif (self.filter_type == self.SIMILARITY_THRESHOLD):
                        filter_type = "Similarity Threshold"
                    else:
                        filter_type = "Combined Top-K & Similarity Threshold"

                    self.logger.info("File: %s", self.path)
                    self.logger.info("Logger file: %s", self.output_file)
                    self.logger.info("Algorithm: %s", 'Item-Based' if self.algorithm == self.ITEM_BASED_ALGORITHM else 'User-Based')
                    self.logger.info("Filter Type: %s", filter_type)
                    self.logger.info("Neighbourhood Size: %d", self.neighbourhood_size)
                    self.logger.info("Similarity Threshold: %f", self.similarity_threshold)
                    self.logger.info("Include Negative Correlations: %s", self.include_negative_correlations)

                    if self.algorithm == self.ITEM_BASED_ALGORITHM:
                        result = self.find_item_mae()
                    elif self.algorithm == self.USER_BASED_ALGORITHM:
                        result = self.find_user_mae()
                    else:
                        self.logger.error("Invalid algorithm selected.")
                        result = None
                    self.logger.info("Minimum filtered neighbourhood size: %f", self.min_filtered_neighbourhood_size)
                    self.logger.info("Maximum filtered neighbourhood size: %f", self.max_filtered_neighbourhood_size)
                    self.logger.info("Minimum neighbours size: %f", self.min_neighbours_size)
                    self.logger.info("Maximum neighbours size: %f", self.max_neighbours_size)
                    self.logger.info("Result: %s\n\n", result)

                return result
        except Exception as e:
            with self.logger_lock:
                self.logger.error("Error in recommender system: %s", e)
            return None
        finally:
            self.close_logger()