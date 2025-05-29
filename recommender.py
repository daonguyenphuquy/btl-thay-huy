import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Lớp triển khai thuật toán khuyến nghị phim"""

    def __init__(self, data_processor, num_factors=50):
        """Khởi tạo với bộ xử lý dữ liệu và số lượng nhân tố tiềm ẩn"""
        self.data_processor = data_processor
        self.num_factors = num_factors
        self.user_ratings_matrix = None
        self.U = None
        self.sigma = None
        self.Vt = None
        self.predicted_ratings = None
        self.build_model()

    def build_model(self):
        """Xây dựng mô hình khuyến nghị sử dụng SVD"""
        print("Building recommendation model...")

        # Tạo ma trận user-item từ dữ liệu đánh giá
        ratings = self.data_processor.ratings

        # Chuyển đổi thành ma trận đánh giá: hàng = users, cột = items
        user_item_df = ratings.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        # Lưu trữ ánh xạ giữa chỉ số hàng/cột và ID người dùng/phim
        self.user_indices = user_item_df.index.tolist()
        self.movie_indices = user_item_df.columns.tolist()

        # Chuyển đổi thành ma trận numpy
        self.user_ratings_matrix = user_item_df.values

        # Chuẩn hóa bằng cách trừ đi đánh giá trung bình của mỗi người dùng
        self.user_ratings_means = np.mean(self.user_ratings_matrix, axis=1)
        ratings_norm = self.user_ratings_matrix - self.user_ratings_means.reshape(-1, 1)

        # Áp dụng SVD để phân rã ma trận
        self.U, self.sigma, self.Vt = svds(ratings_norm, k=self.num_factors)

        # Chuyển đổi sigma từ vector thành ma trận đường chéo
        sigma_diag_matrix = np.diag(self.sigma)

        # Tính toán ma trận đánh giá dự đoán
        self.predicted_ratings = np.dot(np.dot(self.U, sigma_diag_matrix), self.Vt) + self.user_ratings_means.reshape(
            -1, 1)

        print("Recommendation model built successfully!")

    def recommend_for_user(self, user_id, num_recommendations=10):
        """Đưa ra khuyến nghị cho người dùng cụ thể"""
        # Tìm vị trí của người dùng trong ma trận
        if user_id not in self.user_indices:
            # Nếu người dùng không có trong dữ liệu huấn luyện, trả về các phim phổ biến
            return self.data_processor.get_top_rated_movies(num_recommendations)

        user_idx = self.user_indices.index(user_id)

        # Lấy các đánh giá dự đoán cho người dùng này
        user_pred_ratings = self.predicted_ratings[user_idx]

        # Lấy danh sách phim mà người dùng đã đánh giá
        user_rated_movies = self.data_processor.ratings[
            self.data_processor.ratings['userId'] == user_id]['movieId'].tolist()

        # Tạo danh sách các phim mà người dùng chưa đánh giá
        unrated_indices = [i for i, movie_id in enumerate(self.movie_indices)
                           if movie_id not in user_rated_movies]

        # Lấy điểm dự đoán cho các phim chưa đánh giá
        unrated_pred_ratings = [(self.movie_indices[i], user_pred_ratings[i]) for i in unrated_indices]

        # Sắp xếp theo điểm dự đoán giảm dần
        sorted_recommendations = sorted(unrated_pred_ratings, key=lambda x: x[1], reverse=True)

        # Lấy n phim hàng đầu
        top_recommendations = sorted_recommendations[:num_recommendations]

        # Lấy thông tin chi tiết của các phim được khuyến nghị
        recommended_movies = []
        for movie_id, pred_rating in top_recommendations:
            movie_info = self.data_processor.movies[
                self.data_processor.movies['movieId'] == movie_id].iloc[0].to_dict()
            movie_info['predicted_rating'] = round(pred_rating, 2)
            recommended_movies.append(movie_info)

        return recommended_movies

    def get_similar_movies(self, movie_id, num_similar=10):
        """Tìm các phim tương tự dựa trên ma trận đặc trưng từ SVD"""
        # Kiểm tra xem phim có trong dữ liệu huấn luyện không
        if movie_id not in self.movie_indices:
            return []

        # Lấy vị trí của phim trong ma trận
        movie_idx = self.movie_indices.index(movie_id)

        # Lấy vector đặc trưng của phim từ ma trận Vt
        movie_features = self.Vt[:, movie_idx]

        # Tính toán độ tương đồng cosine giữa phim này và tất cả các phim khác
        similarities = cosine_similarity(movie_features.reshape(1, -1), self.Vt.T)

        # Lấy chỉ số của các phim tương tự nhất (bỏ qua phim đầu tiên vì nó chính là phim cần tìm)
        similar_indices = similarities.argsort()[0][::-1][1:num_similar + 1]

        # Lấy thông tin chi tiết của các phim tương tự
        similar_movies = []
        for idx in similar_indices:
            movie_id = self.movie_indices[idx]
            movie_info = self.data_processor.movies[
                self.data_processor.movies['movieId'] == movie_id].iloc[0].to_dict()
            movie_info['similarity_score'] = round(float(similarities[0][idx]), 3)
            similar_movies.append(movie_info)

        return similar_movies