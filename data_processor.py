import os
import pandas as pd
import numpy as np
import requests
import json
import pickle

class DataProcessor:
    """Lớp xử lý dữ liệu MovieLens"""

    def __init__(self, data_path, tmdb_api_key='e4422a03c53edc1876776cad24a6b4c5'):
        """Khởi tạo với đường dẫn đến thư mục dữ liệu MovieLens"""
        print("Initializing DataProcessor...")
        self.data_path = data_path
        self.tmdb_api_key = tmdb_api_key
        self.movies = None
        self.ratings = None
        self.links = None
        self.tags = None
        self.cache_file = os.path.join(data_path, 'processed_data.pkl')
        self.load_or_process_data()
        print("Data processing completed!")

    def load_or_process_data(self):
        """Kiểm tra cache, nếu có thì tải, nếu không thì xử lý và lưu cache"""
        if os.path.exists(self.cache_file):
            print("Loading data from cache...")
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
            self.movies, self.ratings, self.links, self.tags = data
        else:
            print("Cache not found, processing data...")
            self.load_data()
            self.process_data()
            self.save_cache()

    def load_data(self):
        """Tải dữ liệu từ các file CSV"""
        print("Loading data files...")
        movies_path = os.path.join(self.data_path, 'movies.csv')
        ratings_path = os.path.join(self.data_path, 'ratings.csv')
        links_path = os.path.join(self.data_path, 'links.csv')
        tags_path = os.path.join(self.data_path, 'tags.csv')

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.links = pd.read_csv(links_path)
        self.tags = pd.read_csv(tags_path)
        print("All data files loaded.")

    def get_poster_url(self, tmdb_id):
        """Lấy URL ảnh poster từ TMDb dựa trên tmdbId."""
        cache_file = os.path.join(self.data_path, 'poster_cache.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

        tmdb_id_str = str(tmdb_id)
        if tmdb_id_str in cache:
            print(f"Using cached poster for tmdb_id {tmdb_id}")
            return cache[tmdb_id_str]

        if not self.tmdb_api_key or not tmdb_id:
            print(f"No API key or tmdb_id for {tmdb_id}, returning None.")
            return None
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={self.tmdb_api_key}&language=vi"
        print(f"Requesting poster for tmdb_id {tmdb_id}...")
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                poster_url = f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
                print(f"Poster URL for tmdb_id {tmdb_id}: {poster_url}")
                cache[tmdb_id_str] = poster_url
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
                return poster_url
            print(f"No poster_path for tmdb_id {tmdb_id}.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi lấy poster cho tmdb_id {tmdb_id}: {e}")
            return None

    def process_data(self):
        """Xử lý dữ liệu: làm sạch, biến đổi và tạo các features mới"""
        print("Extracting year from movie titles...")
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)$')

        print("Splitting genres...")
        self.movies['genres_list'] = self.movies['genres'].str.split('|')

        print("Merging with links to get tmdbId...")
        self.movies = self.movies.merge(self.links[['movieId', 'tmdbId']], on='movieId', how='left')

        print("Fetching poster URLs...")
        self.movies['poster_url'] = self.movies['tmdbId'].apply(
            lambda x: self.get_poster_url(int(x)) if pd.notna(x) else None
        )

        print("Calculating movie stats...")
        movie_stats = self.ratings.groupby('movieId').agg(
            count=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()

        print("Merging movie stats...")
        self.movies = self.movies.merge(movie_stats, on='movieId', how='left')

        print("Handling missing values...")
        self.movies['count'] = self.movies['count'].fillna(0)
        self.movies['avg_rating'] = self.movies['avg_rating'].fillna(0)
        self.movies['poster_url'] = self.movies['poster_url'].fillna('https://via.placeholder.com/300x450?text=Không+có+ảnh')

    def save_cache(self):
        """Lưu dữ liệu đã xử lý vào file cache"""
        print("Saving processed data to cache...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump((self.movies, self.ratings, self.links, self.tags), f)
        print("Cache saved successfully.")

    def get_top_rated_movies(self, n=10, min_ratings=10):
        """Lấy n phim có đánh giá cao nhất (với ít nhất min_ratings đánh giá)"""
        print("Fetching top rated movies...")
        top_movies = self.movies[self.movies['count'] >= min_ratings].sort_values(
            by=['avg_rating', 'count'], ascending=[False, False]).head(n)
        return top_movies.to_dict('records')

    def get_all_users(self):
        """Lấy danh sách tất cả người dùng từ dữ liệu đánh giá"""
        return sorted(self.ratings['userId'].unique().tolist())

    def get_user_ratings(self, user_id):
        """Lấy tất cả phim mà một người dùng đã đánh giá, kèm thông tin phim"""
        print(f"Fetching ratings for user {user_id}...")
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        user_movies = user_ratings.merge(self.movies, on='movieId')
        user_movies = user_movies.sort_values('timestamp', ascending=False)
        return user_movies.to_dict('records')

    def get_movie_info(self, movie_id):
        """Lấy thông tin chi tiết của một phim dựa trên ID"""
        movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
        return movie.to_dict()

    def get_num_users(self):
        """Số lượng người dùng trong dataset"""
        return self.ratings['userId'].nunique()

    def get_num_movies(self):
        """Số lượng phim trong dataset"""
        return self.movies.shape[0]

    def get_num_ratings(self):
        """Tổng số đánh giá trong dataset"""
        return self.ratings.shape[0]

    def get_avg_rating(self):
        """Đánh giá trung bình của tất cả phim"""
        return self.ratings['rating'].mean()

    def get_rating_distribution(self):
        """Phân phối các đánh giá (bao nhiêu đánh giá cho mỗi điểm từ 0.5-5.0)"""
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        return [{'rating': float(rating), 'count': int(count)}
                for rating, count in rating_counts.items()]

    def get_genre_distribution(self):
        """Phân phối các thể loại phim"""
        genre_data = []
        for _, row in self.movies.iterrows():
            if isinstance(row['genres_list'], list):
                for genre in row['genres_list']:
                    genre_data.append({'movieId': row['movieId'], 'genre': genre})

        genres_df = pd.DataFrame(genre_data)
        genre_counts = genres_df['genre'].value_counts()

        return [{'genre': genre, 'count': int(count)}
                for genre, count in genre_counts.items() if genre != '(no genres listed)']

    def get_movie_data_for_charts(self):
        """Chuẩn bị dữ liệu phim cho trực quan hóa"""
        year_counts = self.movies['year'].value_counts().sort_index()
        year_data = [{'year': str(year), 'count': int(count)}
                     for year, count in year_counts.items() if pd.notna(year)]

        scatter_data = self.movies[['movieId', 'title', 'count', 'avg_rating']]
        scatter_data = scatter_data[scatter_data['count'] > 0]
        scatter_points = [{'id': int(row['movieId']),
                           'title': row['title'],
                           'count': int(row['count']),
                           'rating': float(row['avg_rating'])}
                          for _, row in scatter_data.iterrows()]

        return {
            'year_distribution': year_data,
            'movie_ratings_scatter': scatter_points
        }