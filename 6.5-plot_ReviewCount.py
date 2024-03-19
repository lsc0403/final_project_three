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

# 计算每部电影每个月和每年的评论总数
def calculate_reviews(data):
    review_data = []
    for movie, reviews in data.items():
        for review in reviews:
            date = datetime.strptime(review['Date'], '%Y-%m-%d')
            month = date.strftime('%Y-%m')
            year = date.strftime('%Y')
            review_data.append({'Movie': movie, 'Month': month, 'Year': year})
    df = pd.DataFrame(review_data)
    monthly_reviews = df.groupby(['Movie', 'Month']).size().reset_index(name='TotalReviews')
    yearly_reviews = df.groupby(['Movie', 'Year']).size().reset_index(name='TotalReviews')
    total_reviews_by_movie = df.groupby('Movie').size().reset_index(name='TotalReviews')
    return monthly_reviews, yearly_reviews, total_reviews_by_movie

# 生成折线图
def generate_plot(monthly_reviews, yearly_reviews, total_reviews_by_movie):
    fig = go.Figure()

    # 为每部电影按月和按年添加数据轨迹
    for reviews_data, suffix in [(monthly_reviews, " (Monthly)"), (yearly_reviews, " (Yearly)")]:
        for movie in reviews_data['Movie'].unique():
            movie_data = reviews_data[reviews_data['Movie'] == movie]
            time_period = 'Year' if " (Yearly)" in suffix else 'Month'
            total_reviews = total_reviews_by_movie[total_reviews_by_movie['Movie'] == movie]['TotalReviews'].iloc[0]
            movie_label = f"{movie}{suffix} ({total_reviews} reviews)"

            # 为每个点生成悬停文本
            hover_texts = [f"{period}: {reviews} reviews" for period, reviews in zip(movie_data[time_period], movie_data['TotalReviews'])]

            trace = go.Scatter(
                x=movie_data[time_period],
                y=movie_data['TotalReviews'],
                mode='lines+markers',
                name=movie_label,
                text=hover_texts,  # 指定悬停文本
                hoverinfo='text',  # 设置悬停时显示文本信息
                visible='legendonly'
            )
            fig.add_trace(trace)

    # 添加下拉菜单以切换显示模式
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Yearly",
                         method="update",
                         args=[{"visible": ["legendonly" if " (Yearly)" in t.name else False for t in fig.data]},
                               {"title": "Total reviews per year"}]),
                    dict(label="Monthly",
                         method="update",
                         args=[{"visible": ["legendonly" if " (Monthly)" in t.name else False for t in fig.data]},
                               {"title": "Total reviews per month"}]),
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.8,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ],
        title='Movie Reviews Over Time',
        xaxis_title='Time',
        yaxis_title='Number of Reviews'
    )

    # 导出为HTML
    plot(fig, filename='Page_and_JSON/6.5-plot_ReviewCount.html', auto_open=False, include_plotlyjs='cdn')

if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'  # 确保这里的路径与您的文件路径一致
    data = load_data(filepath)
    monthly_reviews, yearly_reviews, total_reviews_by_movie = calculate_reviews(data)
    generate_plot(monthly_reviews, yearly_reviews, total_reviews_by_movie)