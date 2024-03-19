import json
import pandas as pd
from json import JSONEncoder
from collections import defaultdict
from datetime import datetime
import plotly.graph_objs as go
from plotly.offline import plot

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Period):
            return str(obj)
        return JSONEncoder.default(self, obj)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_average_ratings(data):
    ratings_by_month = defaultdict(lambda: defaultdict(list))
    overall_average_ratings = {}
    review_counts = defaultdict(lambda: defaultdict(int))
    total_reviews_by_movie = {}
    for movie, reviews in data.items():
        total_ratings = 0
        total_reviews = 0
        for review in reviews:
            date = datetime.strptime(review['Date'], '%Y-%m-%d')
            month = date.strftime('%Y-%m')
            ratings_by_month[movie][month].append(review['Rating'])
            review_counts[movie][month] += 1
            total_ratings += review['Rating']
            total_reviews += 1
        overall_average_ratings[movie] = total_ratings / total_reviews if total_reviews else 0
        total_reviews_by_movie[movie] = total_reviews
    average_ratings = {movie: {month: sum(ratings) / len(ratings) for month, ratings in months.items()} for movie, months in ratings_by_month.items()}
    return average_ratings, review_counts, overall_average_ratings, total_reviews_by_movie


def generate_plot(average_ratings, review_counts, overall_average_ratings, total_reviews_by_movie):
    fig = go.Figure(layout=dict(height=1000))  # 设置图表宽度为 1000 像素


    # 按月数据的轨迹
    for movie, ratings in sorted(average_ratings.items()):
        months = sorted(ratings.keys())
        avg_ratings = [ratings[month] for month in months]
        counts = [review_counts[movie][month] for month in months]
        hover_texts = [f"{month}: {rating:.1f} stars, {count} reviews" for month, rating, count in
                       zip(months, avg_ratings, counts)]
        # 在电影名旁边添加总评论数和总平均星级
        movie_label_with_reviews = f"{movie} (Monthly) ({total_reviews_by_movie[movie]} reviews, {overall_average_ratings[movie]:.1f} stars)"
        trace = go.Scatter(x=months, y=avg_ratings, mode='lines+markers', name=movie_label_with_reviews, text=hover_texts,
                           hoverinfo='text+name', visible='legendonly')
        fig.add_trace(trace)

    # 按年数据的轨迹（重新计算）
    for movie, ratings in sorted(average_ratings.items()):
        years = sorted(set(month[:4] for month in ratings.keys()))
        avg_ratings_yearly = [sum(ratings[month] for month in ratings.keys() if month.startswith(year)) / len(
            [month for month in ratings.keys() if month.startswith(year)]) for year in years]
        counts_yearly = [
            sum(review_counts[movie][month] for month in review_counts[movie].keys() if month.startswith(year)) for year
            in years]
        hover_texts_yearly = [f"{year}: {rating:.1f} stars, {count} reviews" for year, rating, count in
                              zip(years, avg_ratings_yearly, counts_yearly)]
        # 在电影名旁边添加总评论数和总平均星级
        movie_label_with_reviews_yearly = f"{movie} (Yearly) ({total_reviews_by_movie[movie]} reviews, {overall_average_ratings[movie]:.1f} stars)"
        trace_yearly = go.Scatter(x=years, y=avg_ratings_yearly, mode='lines+markers', name=movie_label_with_reviews_yearly,
                                  text=hover_texts_yearly, hoverinfo='text+name', visible='legendonly')
        fig.add_trace(trace_yearly)

    # 添加下拉菜单以切换视图
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Yearly",
                         method="update",
                         args=[{"visible": ["legendonly" if " (Yearly)" in t.name else False for t in fig.data]},
                               {"title": "Average Movie Ratings Over Time (Yearly)"}]),
                    dict(label="Monthly",
                         method="update",
                         args=[{"visible": ["legendonly" if " (Monthly)" in t.name else False for t in fig.data]},
                               {"title": "Average Movie Ratings Over Time (Monthly)"}]),
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.1,  # 调整位置，避免与标题重叠
                xanchor="right",
                y=1.15,  # 位于图表上方外部
                yanchor="top"

            ),
        ],
        title='Average Movie Ratings Over Time',
        title_y=0.05,
        xaxis=dict(title='Time'),
        yaxis=dict(title='Average Rating'),
        legend=dict(title='Legend', orientation="h", x=0, y=1.1, xanchor='left', yanchor='bottom')
    )

    plot(fig, filename='Page_and_JSON/6.3-plot_star.html', auto_open=False, include_plotlyjs='cdn')


if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'  # 确保文件路径正确
    data = load_data(filepath)
    average_ratings, review_counts, overall_average_ratings, total_reviews_by_movie = calculate_average_ratings(data)
    generate_plot(average_ratings, review_counts, overall_average_ratings, total_reviews_by_movie)
