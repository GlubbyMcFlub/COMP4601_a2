import os
import math
import numpy as np
import time

class RecommenderSystem:
    '''
    RecommenderSystem class for COMP4601 Assignment 2.
    '''
    MISSING_RATING = 0
    DEFAULT_NEIGHBOURHOOD_SIZE = 5
    DEFAULT_SIMILARITY_THRESHOLD = 0.5
    ITEM_BASED_ALGORITHM = 0
    USER_BASED_ALGORITHM = 1
    TOP_K_NEIGHBOURS = 0
    SIMILARITY_THRESHOLD = 1
    MIN_RATING = 1.0
    MAX_RATING = 5.0

    def __init__(self, path, algorithm, neighbourhood_size=DEFAULT_NEIGHBOURHOOD_SIZE, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, include_negative_correlations=False, filter_type=TOP_K_NEIGHBOURS):
        """
        Initialize the RecommenderSystem object.

        Parameters:
        - path (str): Path to the input data file.
        - neighbourhood_size (int): Size of the neighbourhood for recommendation.
        - neighbourhood_threshold (float): Threshold for including user/item in neighbourhood.
        - include_negative_correlations (bool): Whether to include negative correlations in recommendations.

        Returns:
        None
        """
        self.path = path
        self.algorithm = algorithm
        self.neighbourhood_size = neighbourhood_size
        self.similarity_threshold = similarity_threshold
        self.include_negative_correlations = include_negative_correlations
        self.filter_type = filter_type
        self.num_users, self.num_items, self.users, self.items, self.ratings = self.read_data()

    def read_data(self):
        """
        Read data from the input file and parse user-item ratings.

        Returns:
        Tuple containing:
        - num_users (int): Number of users in the dataset.
        - num_items (int): Number of items in the dataset.
        - users (numpy.ndarray): Array containing user labels.
        - items (numpy.ndarray): Array containing item labels.
        - ratings (numpy.ndarray): 2D array containing user-item ratings.
        """
        try:
            with open(self.path, 'r') as file:
                num_users, num_items = map(int, file.readline().split())
                users = np.array(file.readline().split())
                items = np.array(file.readline().split())
                ratings = np.array([[float(rating) if rating != self.MISSING_RATING else self.MISSING_RATING for rating in line.split()] for line in file])
                return num_users, num_items, users, items, ratings
        except ValueError as err:
            raise ValueError(f"Error: {str(err)}") from err
        except FileNotFoundError as err:
            print(f"Error: {str(err)}")
            return None, None, None, None, None


    def compute_user_similarity(self, user1_ratings, user2_ratings, common_items):
        """
        Computes the Pearson correlation coefficient (PCC) between two users based on their common item ratings.

        Parameters:
        - user1_ratings (list): Ratings of user 1 for common items.
        - user2_ratings (list): Ratings of user 2 for common items.
        - common_items (list): List of indices for common items.

        Returns:
        - correlation (float): Pearson correlation coefficient.
        """

        num_common_items = len(common_items)

        if num_common_items == 0:
            return 0

        user1_ratings_common = [user1_ratings[i] for i in common_items]
        user2_ratings_common = [user2_ratings[i] for i in common_items]
        user1_ratings_all = [x for x in user1_ratings if x != self.MISSING_RATING]
        user2_ratings_all = [x for x in user2_ratings if x != self.MISSING_RATING]

        mean_user1 = np.mean(user1_ratings_all)
        mean_user2 = np.mean(user2_ratings_all)

        numerator = sum((user1_ratings_common[i] - mean_user1) * (user2_ratings_common[i] - mean_user2) for i in range(num_common_items))
        denominator_user1 = math.sqrt(sum((user1_ratings_common[i] - mean_user1) ** 2 for i in range(num_common_items)))
        denominator_user2 = math.sqrt(sum((user2_ratings_common[i] - mean_user2) ** 2 for i in range(num_common_items)))

        if denominator_user1 * denominator_user2 == 0:
            return 0

        correlation = numerator / (denominator_user1 * denominator_user2)

        if self.include_negative_correlations:
            return correlation
        else:
            return max(0, correlation)

    def compute_item_similarity(self, item1_index, item2_index, common_users, average_ratings):
        """
        Compute the similarity between two items based on common user ratings.

        Parameters:
        - item1_index (int): Index of the first item.
        - item2_index (int): Index of the second item.
        - common_users (numpy.ndarray): Array of indices for users who have rated both items.
        - average_ratings (numpy.ndarray): Array of average ratings for each user.

        Returns:
        float: Similarity between the two items.
        """
        
        num_common_users = len(common_users)

        if num_common_users == 0:
            return 0
        
        
        # numerator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) * (self.ratings[common_users, item2_index] - average_ratings[common_users]))
        # item1_denominator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) ** 2)
        # item2_denominator = np.sum((self.ratings[common_users, item2_index] - average_ratings[common_users]) ** 2)
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
        # print(f"{item1_index}, {item2_index}, {numerator}, {denominator}")
        similarity = max(0, numerator / denominator) if denominator != 0 else 0

        return similarity

    def precompute_item_similarities(self, itemIndex, average_ratings):
        """
        Precompute item similarities for the given item.

        Returns:
        numpy.ndarray: array containing precomputed item similarities.
        """
        
        similarities = np.zeros((self.num_items, self.num_items), dtype=np.float64)

        item1_ratings = np.where(self.ratings[:, itemIndex] != self.MISSING_RATING)[0]
        
        for j in range(self.num_items):
            if j != itemIndex:
                # find common users who rated both items itemIndex and j
                item2_ratings = np.where(self.ratings[:, j] != self.MISSING_RATING)[0]
                common_users = np.intersect1d(item1_ratings, item2_ratings)

                # calculate similarity between items itemIndex and j based on common users
                similarity = self.compute_item_similarity(itemIndex, j, common_users, average_ratings)
                similarities[itemIndex, j] = similarity
                similarities[j, itemIndex] = similarity

        return similarities
    
    def precompute_user_similarities(self):
        """
        Precomputes the similarities between all pairs of users.

        Returns:
        - similarities (numpy.array): Matrix of precomputed similarities.
        """
        
        similarities = np.zeros((self.num_users, self.num_users), dtype=np.float64)

        for i in range(self.num_users):
            # Calculate for unique pairs
            for j in range(i + 1, self.num_users): 
                user1_indices = np.where(self.ratings[i] != self.MISSING_RATING)[0]
                user2_indices = np.where(self.ratings[j] != self.MISSING_RATING)[0]

                intersecting_indices = np.intersect1d(user1_indices, user2_indices)
                common_items = intersecting_indices

                similarity = self.compute_user_similarity(self.ratings[i], self.ratings[j], common_items)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

        return similarities
    
    def predict_ratings(self, similarities):
        #currently just here from lab#6
        """
        Predicts the ratings for items that a user has not rated yet, using collaborative filtering.

        Parameters:
        - similarities (numpy.array): Matrix of precomputed similarities.

        Returns:
        - predicted_ratings (list): List of predicted ratings for each user.
        """
        
        predicted_ratings = []

        for i in range(self.num_users):
            current_user_ratings = self.ratings[i]
            current_user_predicted_ratings = np.full(self.num_items, self.MISSING_RATING, dtype=float)

            for j in range(self.num_items):
                if current_user_ratings[j] == self.MISSING_RATING:
                    neighbours = []

                    for k in range(self.num_users):
                        if i != k and self.ratings[k, j] != self.MISSING_RATING:
                            neighbours.append((k, similarities[i, k]))

                    neighbours.sort(key=lambda x: x[1], reverse=True)
                    top_neighbours = neighbours[:self.neighbourhood_size]
                    # TODO: Could also use a threshold for similarity
                    sum_ratings = 0
                    total_similarity = 0

                    for neighbour_index, similarity in top_neighbours:
                        neighbour_rating = self.ratings[neighbour_index, j]
                        # Calculate the deviation from the average
                        filtered_ratings = [x for x in self.ratings[neighbour_index] if x != self.MISSING_RATING]
                        deviation = neighbour_rating - np.mean(filtered_ratings)
                        # Accumulate the weighted sum of deviations
                        sum_ratings += similarity * deviation
                        total_similarity += similarity

                    # Calculate predicted rating
                    user_average = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])
                    influence = sum_ratings / total_similarity
                    predicted_rating = user_average + influence
                    # Ensure rating is between 0 and 5
                    current_user_predicted_ratings[j] = max(0, min(5, predicted_rating))

            predicted_ratings.append(current_user_predicted_ratings)

        return predicted_ratings

    def predict_rating(self, userIndex, itemIndex, similarities):
        #TODO: choose top neighbours based on threshold instead
        """
        Predict a single rating for a user-item pair.

        Parameters:
        - userIndex (int): Index of the user.
        - itemIndex (int): Index of the item.
        - similarities (numpy.ndarray): 2D array containing precomputed item similarities.

        Returns:
        Tuple containing:
        - predict_rating (float): Predicted rating for the user-item pair.
        - total_similarity (float): Total similarity score of the neighbourhood.
        - adjusted_neighbourhood_size (int): Adjusted size of the neighbourhood used for prediction.
        """
        
        try:
            # get all neighbours who rated the item, adjust neighbourhood size if necessary
            neighbours = np.where((self.ratings[userIndex] != self.MISSING_RATING) & (similarities[itemIndex] > 0))[0]
            adjusted_neighbourhood_size = min(self.neighbourhood_size, len(neighbours))

            # if no neighbours found, use average rating for item
            if adjusted_neighbourhood_size == 0:
                ratings_without_zeros = np.where(self.ratings[userIndex] != 0, self.ratings[userIndex], np.nan)
                predict_rating = np.nanmean(ratings_without_zeros)

                total_similarity = 0
            else:
                #create array of tuples using neighbours indices and similarities
                # TODO: VERIFY THAT THIS IS CORRECT!!!!!!!!!!!!!!!!!
                if (self.include_negative_correlations or self.filter_type == self.TOP_K_NEIGHBOURS):
                    neighbourhood_similarities = [(num, i, i in neighbours) for i, num in enumerate(abs(similarities[itemIndex]))]
                else:
                    neighbourhood_similarities = [(num, i, i in neighbours) for i, num in enumerate(similarities[itemIndex])]

                #sort array based on similarities in descending order for neighbourhood similarities
                sorted_indices = np.array([index for _, index, is_in_array in sorted(neighbourhood_similarities, key=lambda x: (x[0], -x[1]), reverse=True) if is_in_array])
                
                # Filtering technique
                if (self.filter_type == self.SIMILARITY_THRESHOLD):
                    filtered_indices = sorted_indices[sorted_indices > self.similarity_threshold]
                elif (self.filter_type == self.TOP_K_NEIGHBOURS):
                    filtered_indices = sorted_indices[:adjusted_neighbourhood_size]
                else:
                    filtered_indices = sorted_indices[sorted_indices > self.similarity_threshold][:adjusted_neighbourhood_size]
                    
                #print(f"Filtered indices: {filtered_indices}")

                sum_ratings = np.sum(similarities[itemIndex, filtered_indices] * self.ratings[userIndex, filtered_indices])
                if (self.include_negative_correlations):
                    total_similarity = np.sum(abs(similarities[itemIndex, filtered_indices]))
                else:
                    total_similarity = np.sum(similarities[itemIndex, filtered_indices])
                

                predict_rating = max(0, min(5, sum_ratings / total_similarity)) if total_similarity != 0 else np.nanmean(self.ratings[userIndex])

            return predict_rating, total_similarity, adjusted_neighbourhood_size
        except Exception as err:
            print(f"Error in predict_rating: {str(err)}")
            return None, None, None
    
    def find_similarity_threshold_mae(self):
        """
        Find the Mean Absolute Error (MAE) for the similarity threshold-based prediction model.

        Returns:
        float: Mean Absolute Error (MAE) for the similarity threshold-based prediction model.
        """
        
        try:
            start_time = time.time()
            test_set_size = 0
            numerator = 0

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0

            similarities = self.precompute_item_similarities()

            for i in range(self.num_users):
                for j in range(self.num_items):
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        test_set_size += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING

                        # predict the rating for each user-item pair
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_rating(i, j, similarities)

                        if not np.isnan(predicted_rating):
                            error = abs(predicted_rating - temp)
                            numerator += error

                            if error < self.MIN_RATING:
                                under_predictions += 1
                            elif error > self.MAX_RATING:
                                over_predictions += 1

                        if np.isnan(predicted_rating) or np.isinf(predicted_rating) or total_similarity == 0 or total_similarity < self.similarity_threshold:
                            no_valid_neighbours += 1
                        else:
                            total_neighbours_used += adjusted_neighbourhood_size

                        self.ratings[i, j] = temp

            mae = numerator / test_set_size
            print(f"MAE: {mae}")

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                minutes, seconds = divmod(elapsed_time, 60)
                print(f"Elapsed time: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
            else:
                print(f"Elapsed time: {elapsed_time:.3f}s")

            return mae
        except Exception as err:
            print(f"Error: {str(err)}")
            return None

    def find_user_mae(self):
        """
        Find the Mean Absolute Error (MAE) for the user-based prediction model.

        Returns:
        float: Mean Absolute Error (MAE) for the user-based prediction model.
        """
        
        try:
            start_time = time.time()
            test_set_size = 0
            numerator = 0

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0

            similarities = self.precompute_user_similarities()

            for i in range(self.num_users):
                for j in range(self.num_items):
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        test_set_size += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING

                        # predict the rating for each user-item pair
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_rating(i, j, similarities)
                        #print(f"Predicted rating: {predicted_rating}, total similarity: {total_similarity}, adjusted neighbourhood size: {adjusted_neighbourhood_size}")

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

            mae = numerator / test_set_size
            print(f"MAE: {mae}")

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                minutes, seconds = divmod(elapsed_time, 60)
                print(f"Elapsed time: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
            else:
                print(f"Elapsed time: {elapsed_time:.3f}s")

            return mae
        except Exception as err:
            print(f"Error: {str(err)}")
            return None

    def find_item_mae(self):
        """
        Find the Mean Absolute Error (MAE) for the prediction model.

        Returns:
        float: Mean Absolute Error (MAE) for the prediction model.
        """
        
        try:
            start_time = time.time()
            test_set_size = 0
            numerator = 0
            similarity_time = 0
            prediction_time = 0

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0
            # calculate average ratings for all users, ignoring any missing ratings
            average_ratings = np.nanmean(np.where(self.ratings != self.MISSING_RATING, self.ratings, np.nan), axis=1)

            for i in range(self.num_users):
                for j in range(self.num_items):
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        
                        test_set_size += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING
                        average_ratings[i] = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])

                        similarity_start_time = time.time()
                        similarities = self.precompute_item_similarities(j, average_ratings)
                        similarity_time += time.time() - similarity_start_time

                        # predict the rating for each user-item pair
                        prediction_start_time = time.time()
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_rating(i, j, similarities)
                        prediction_time += time.time() - prediction_start_time

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
            # print(f"Numerator = {numerator}")
            # print(f"test_set_size = {test_set_size}")
            # print(f"Total predictions: {test_set_size}")
            # print(f"Total under predictions (< {1}): {under_predictions}")
            # print(f"Total over predictions (> {5}): {over_predictions}")
            # print(f"Number of cases with no valid neighbours: {no_valid_neighbours}")
            # print(f"Average neighbours used: {total_neighbours_used / test_set_size}")
            print(f"MAE: {mae}")

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                minutes, seconds = divmod(elapsed_time, 60)
                print(f"Start: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
            else:
                print(f"Start: {elapsed_time:.3f}s")
            if similarity_time >= 60:
                minutes, seconds = divmod(similarity_time, 60)
                print(f"similarity: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
            else:
                print(f"similarity: {elapsed_time:.3f}s")
            if prediction_time >= 60:
                minutes, seconds = divmod(prediction_time, 60)
                print(f"prediction_time: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
            else:
                print(f"prediction_time: {elapsed_time:.3f}s")

            return mae
        except Exception as err:
            print(f"Error: {str(err)}")
            return None

        
    def run(self):
        '''
        Run the recommender system.

        Returns:
        float: Result of the algorithm (e.g., MAE).
        '''
        print("Running the recommender system...")
        print(f"Algorithm: {'Item-Based' if self.algorithm == self.ITEM_BASED_ALGORITHM else 'User-Based'}")
        print(f"Filter Type: {'Top-K Neighbours' if self.filter_type == self.TOP_K_NEIGHBOURS else 'Similarity Threshold'}")
        print(f"Neighbourhood Size: {self.neighbourhood_size}")
        print(f"Similarity Threshold: {self.similarity_threshold}")
        print(f"Include Negative Correlations: {self.include_negative_correlations}")

        if self.algorithm == self.ITEM_BASED_ALGORITHM:
            result = self.find_item_mae()
        elif self.algorithm == self.USER_BASED_ALGORITHM:
            result = self.find_user_mae()
        else:
            print("Invalid algorithm selected.")
            result = None

        print(f"Result: {result}")
        
        return result

def main():
    #TODO: add parameter options for item/user and top k neighbours/threshold
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    try:
        print("Select a file to process:")
        for index, file in enumerate(files):
            print(f"{index + 1}. {file}")
        selected_index = int(input("File to process: ")) - 1
        selected_file = os.path.join(input_directory, files[selected_index])

        print("\nAlgorithm - item or user?:")
        print("1. item") # implemented
        print("2. user") # TODO: implement (changing the algorithm will change the similarity function) -> lab 06 extension, optimize MAE?
        selected_index = int(input("Algorithm: ")) - 1
        selected_algorithm = selected_index

        print("\nParameter - top k neighbours or similarity threshold?:")
        print("1. top k neighbours") # implemented
        print("2. similarity threshold") # TODO: implement -> experiment w/ values, neighbourhood size
        selected_index = int(input("Parameter: ")) - 1
        selected_parameter = selected_index
        
        recommender_system = RecommenderSystem(selected_file, selected_algorithm, selected_parameter)
        if recommender_system.algorithm == recommender_system.ITEM_BASED_ALGORITHM:
            recommender_system.find_item_mae()
        else:
            #TODO: replace with user algorithm
            recommender_system.find_item_mae()
            
        '''
        TODO:
            - optimize item-based (Eric to start working on this)
            - implement user based algorithm (David to work on this tomorrow) -> (Eric to start optimizing this tomorrow evening)
            - similarity threshold (David to work on this tomorrow) -> (Eric to start optimizing this tomorrow evening)
            - negative correlations (top-k based on ABS of correlation) -> create option during main (David to start working on this)
            - experiment w/ neighbourhood size, threshold, top-k, user/item based permutations.
            - near the end, implement some sort of test suite (David to start working on this)
        '''
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

if __name__ == "__main__":
    main()