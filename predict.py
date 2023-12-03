import os
import math
import numpy as np
import time

class RecommenderSystem:
    MISSING_RATING = 0.0
    DEFAULT_NEIGHBOURHOOD_SIZE = 5
    DEFAULT_NEIGHBOURHOOD_THRESHOLD = 0.5
    MIN_RATING = 1.0
    MAX_RATING = 5.0

    def __init__(self, path, neighbourhood_size=DEFAULT_NEIGHBOURHOOD_SIZE, neighbourhood_threshold=DEFAULT_NEIGHBOURHOOD_THRESHOLD):
        """
        Initialize the RecommenderSystem object.

        Parameters:
        - path (str): Path to the input data file.
        - neighbourhood_size (int): Size of the neighbourhood for recommendation.
        - neighbourhood_threshold (float): Threshold for including user/item in neighbourhood.

        Returns:
        None
        """
        self.path = path
        self.neighbourhood_size = neighbourhood_size
        self.neighbourhood_threshold = neighbourhood_threshold
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
            raise ValueError(f"Error: {str(err)}")
        except Exception as err:
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
        return correlation
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

                similarity = self.compute_similarity(self.ratings[i], self.ratings[j], common_items)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

        return similarities

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

        numerator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) * (self.ratings[common_users, item2_index] - average_ratings[common_users]))
        item1_denominator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) ** 2)
        item2_denominator = np.sum((self.ratings[common_users, item2_index] - average_ratings[common_users]) ** 2)

        denominator = math.sqrt(item1_denominator * item2_denominator)
        similarity = max(0, numerator / denominator) if denominator != 0 else 0

        return similarity

    def precompute_item_similarities(self, itemIndex):
        """
        Precompute item similarities for the entire dataset.

        Returns:
        numpy.ndarray: array containing precomputed item similarities.
        """
        
        similarities = np.zeros((self.num_items, self.num_items), dtype=np.float64)
        # calculate average ratings for all users, ignoring any missing ratings
        average_ratings = np.nanmean(np.where(self.ratings != self.MISSING_RATING, self.ratings, np.nan), axis=1)

        for j in range(self.num_items):
            if j != itemIndex:
                # find common users who rated both items i and j
                item1_ratings = np.where(self.ratings[:, itemIndex] != self.MISSING_RATING)[0]
                item2_ratings = np.where(self.ratings[:, j] != self.MISSING_RATING)[0]
                common_users = np.intersect1d(item1_ratings, item2_ratings)
                # calculate ismilarity between items i and j based on common users
                similarity = self.compute_item_similarity(itemIndex, j, common_users, average_ratings)
                similarities[itemIndex, j] = similarity
                similarities[j, itemIndex] = similarity

        return similarities

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
        
        # print(f"\nPredicting for user: {self.users[userIndex]}")
        # print(f"Predicting for item: {self.items[itemIndex]}")

        # get all neighbours who rated the item, adjust neighbourhood size if necessary
        neighbours = np.where((self.ratings[userIndex] != self.MISSING_RATING) & (similarities[itemIndex] > 0))[0]
        adjusted_neighbourhood_size = min(self.neighbourhood_size, len(neighbours))

        # if no neighbours found, use average rating for item
        if adjusted_neighbourhood_size == 0:
            ratings_without_zeros = np.where(self.ratings[userIndex] != 0, self.ratings[userIndex], np.nan)
            predict_rating = np.nanmean(ratings_without_zeros)

            total_similarity = 0
            # print("No valid neighbours found.")
        else:
            # print(f"Found {adjusted_neighbourhood_size} valid neighbours:")

            #create array of tuples using neighbours indices and similarities 
            neighbourhood_similarities = [(num, i, i in neighbours) for i, num in enumerate(similarities[itemIndex])]

            #sort array based on similarities in descending order for neighbourhood similarities
            sorted_indices = [index for _, index, is_in_array in sorted(neighbourhood_similarities, key=lambda x: (x[0], -x[1]), reverse=True) if is_in_array]

            top_neighbours_indices = sorted_indices[:adjusted_neighbourhood_size]

            sum_ratings = np.sum(similarities[itemIndex, top_neighbours_indices] * self.ratings[userIndex, top_neighbours_indices])
            total_similarity = np.sum(similarities[itemIndex, top_neighbours_indices])

            # print(f"Initial predicted value: {sum_ratings / total_similarity}")
            predict_rating = max(0, min(5, sum_ratings / total_similarity)) if total_similarity != 0 else np.nanmean(self.ratings[userIndex])

            # print(f"Final predicted value: {predict_rating}")

        return predict_rating, total_similarity, adjusted_neighbourhood_size

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

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0

            for i in range(self.num_users):
                for j in range(self.num_items):
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        test_set_size += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING

                        similarities = self.precompute_item_similarities(j)

                        # predict the rating for each user-item pair
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_rating(i, j, similarities)

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

            return mae
        except Exception as err:
            print(f"Error: {str(err)}")
            return None

def main():
    #TODO: add parameter options for item/user and top k neighbours/threshold
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    print("Select a file to process:")
    for index, file in enumerate(files):
        print(f"{index + 1}. {file}")

    try:
        selected_index = int(input("File to process: ")) - 1
        selected_file = os.path.join(input_directory, files[selected_index])
        recommender_system = RecommenderSystem(selected_file)
        recommender_system.find_item_mae()
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

if __name__ == "__main__":
    main()
