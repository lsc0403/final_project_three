import json
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from plotly.offline import plot

# 读取并解析JSON数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 计算每部电影每个月的正向情感评价率及评论总数，并计算每部电影的总评论数和总正向情感评价率
def calculate_positive_sentiment_rate(data):
    sentiment_data = []
    for movie, reviews in data.items():
        for review in reviews:
            date = datetime.strptime(review['Date'], '%Y-%m-%d')
            month = date.strftime('%Y-%m')
            year = date.strftime('%Y')  # 新增：计算年份
            sentiment_data.append({
                'Movie': movie,
                'Month': month,
                'Year': year,  # 新增
                'PredictedClass': review['Predicted Class']
            })
    df = pd.DataFrame(sentiment_data)
    df['PositiveSentiment'] = df['PredictedClass'].apply(lambda x: 1 if x == 0 else 0)
    # 按月计算
    monthly_sentiment = df.groupby(['Movie', 'Month']).agg({
        'PositiveSentiment': 'mean',
        'PredictedClass': 'count'
    }).rename(columns={'PredictedClass': 'TotalReviews'}).reset_index()
    # 按年计算
    yearly_sentiment = df.groupby(['Movie', 'Year']).agg({
        'PositiveSentiment': 'mean',
        'PredictedClass': 'count'
    }).rename(columns={'PredictedClass': 'TotalReviews'}).reset_index()
    # 计算总评价
    total_reviews_by_movie = df.groupby('Movie')['PredictedClass'].count().reset_index().rename(columns={'PredictedClass': 'TotalReviews'})
    total_sentiment_by_movie = df.groupby('Movie')['PositiveSentiment'].mean().reset_index().rename(columns={'PositiveSentiment': 'TotalSentimentRate'})
    return monthly_sentiment, yearly_sentiment, total_reviews_by_movie, total_sentiment_by_movie

# 生成折线图，电影名后面加上影评总数和总正向情感评价率
def generate_plot(monthly_sentiment, yearly_sentiment, total_reviews_by_movie, total_sentiment_by_movie):
    fig = go.Figure()

    # 添加按月和按年的数据，所有数据初始状态设置为 'legendonly'
    for sentiment_data, suffix in [(monthly_sentiment, " (Monthly)"), (yearly_sentiment, " (Yearly)")]:
        for movie in sentiment_data['Movie'].unique():
            movie_data = sentiment_data[sentiment_data['Movie'] == movie]
            time_period = 'Year' if " (Yearly)" in suffix else 'Month'
            total_reviews = total_reviews_by_movie[total_reviews_by_movie['Movie'] == movie]['TotalReviews'].iloc[0]
            total_sentiment_rate = total_sentiment_by_movie[total_sentiment_by_movie['Movie'] == movie]['TotalSentimentRate'].iloc[0]
            movie_label = f"{movie}{suffix} ({total_reviews} reviews, {total_sentiment_rate:.2f} positive rate)"
            hover_texts = [f"{period}: {rate:.2f} rate, {total} reviews" for period, rate, total in zip(movie_data[time_period], movie_data['PositiveSentiment'], movie_data['TotalReviews'])]
            trace = go.Scatter(x=movie_data[time_period], y=movie_data['PositiveSentiment'], mode='lines+markers',
                               name=movie_label, text=hover_texts, hoverinfo='text', visible='legendonly')
            fig.add_trace(trace)

    # 修改下拉菜单位置，使其位于标题的正右方
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Yearly",
                         method="update",
                         args=[{"visible": ["legendonly" if " (Yearly)" in t.name else False for t in fig.data]},
                               {"title": "The positive sentiment rate of movies over time (yearly)"}]),
                    dict(label="Monthly",
                         method="update",
                         args=[{"visible": ["legendonly" if " (Monthly)" in t.name else False for t in fig.data]},
                               {"title": "The positive sentiment rate of movies over time (monthly)"}]),
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                # 将下拉菜单位置调整到标题的正右方
                x=1.1,
                xanchor="right",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    fig.update_layout(title='The positive sentiment rate of movies over time', xaxis_title='Time', yaxis_title='Positive sentiment rate', legend_title='Movies')
    plot(fig, filename='Page_and_JSON/6.2-plot_sentiment.html', auto_open=False, include_plotlyjs='cdn')

if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'  # 确保这里的路径与您的文件路径一致
    data = load_data(filepath)
    # 更新以接收四个返回值
    monthly_sentiment, yearly_sentiment, total_reviews_by_movie, total_sentiment_by_movie = calculate_positive_sentiment_rate(data)
    generate_plot(monthly_sentiment, yearly_sentiment, total_reviews_by_movie, total_sentiment_by_movie)
