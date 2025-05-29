import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from data_processor import DataProcessor
from recommender import MovieRecommender
import pandas as pd
import json
import secrets
from datetime import datetime

print("Starting application...")  # Thêm dòng này để debug

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Khởi tạo các đối tượng xử lý dữ liệu và khuyến nghị
data_path = os.path.join('data', 'ml-latest-small')
print(f"Loading data from: {data_path}")  # Thêm debug
data_processor = DataProcessor(data_path, tmdb_api_key='e4422a03c53edc1876776cad24a6b4c5')
recommender = MovieRecommender(data_processor)

# Định nghĩa bộ lọc timestamp_to_date
def timestamp_to_date(timestamp):
    """Chuyển đổi timestamp thành định dạng ngày tháng"""
    date = datetime.fromtimestamp(timestamp)
    return date.strftime('%d %b %Y')

# Đăng ký bộ lọc với Jinja2
app.jinja_env.filters['timestamp_to_date'] = timestamp_to_date

@app.route('/')
def index():
    """Trang chủ của ứng dụng"""
    top_movies = data_processor.get_top_rated_movies(10)
    all_users = data_processor.get_all_users()
    return render_template('index.html',
                           top_movies=top_movies,
                           all_users=all_users)

@app.route('/user/<int:user_id>')
def user_profile(user_id):
    """Hiển thị trang profile của người dùng với các phim đã đánh giá"""
    user_ratings = data_processor.get_user_ratings(user_id)
    recommendations = recommender.recommend_for_user(user_id, 10)
    return render_template('recommendations.html',
                           user_id=user_id,
                           user_ratings=user_ratings,
                           recommendations=recommendations)

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint để nhận khuyến nghị phim cho người dùng được chọn"""
    user_id = int(request.form.get('user_id'))
    num_recommendations = int(request.form.get('num_recommendations', 10))
    recommendations = recommender.recommend_for_user(user_id, num_recommendations)
    user_ratings = data_processor.get_user_ratings(user_id)
    return render_template('recommendations.html',
                           user_id=user_id,
                           user_ratings=user_ratings,
                           recommendations=recommendations)

@app.route('/analytics')
def analytics():
    """Trang phân tích và trực quan hóa dữ liệu phim"""
    stats = {
        'num_users': data_processor.get_num_users(),
        'num_movies': data_processor.get_num_movies(),
        'num_ratings': data_processor.get_num_ratings(),
        'avg_rating': data_processor.get_avg_rating()
    }
    rating_distribution = data_processor.get_rating_distribution()
    genre_distribution = data_processor.get_genre_distribution()
    return render_template('analytics.html',
                           stats=stats,
                           rating_distribution=json.dumps(rating_distribution),
                           genre_distribution=json.dumps(genre_distribution))

@app.route('/api/movie_data')
def movie_data():
    """API endpoint để lấy dữ liệu phim cho biểu đồ JavaScript"""
    return jsonify(data_processor.get_movie_data_for_charts())

if __name__ == '__main__':
    app.run(debug=True)