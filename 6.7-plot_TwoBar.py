import json
import plotly.graph_objs as go
from plotly.offline import plot
from collections import defaultdict

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    # 按电影汇总每年的情感分类统计
    sentiment_counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # movie -> year -> [positive, neutral, negative]
    for movie, reviews in data.items():
        for review in reviews:
            year = review['Date'].split("-")[0]  # 假定日期格式为 YYYY-MM-DD
            sentiment = review['Predicted Class']
            sentiment_counts[movie][year][sentiment] += 1
    return sentiment_counts

def create_visualization(sentiment_counts):
    all_movies = list(sentiment_counts.keys())
    dropdown_options = [{"label": movie, "method": "update", "args": [{"visible": [movie == m for m in all_movies for _ in range(3)]}, {"title": f"Sentiment Analysis for {movie}"}]} for movie in all_movies]

    data = []
    for movie, years_data in sentiment_counts.items():
        for i, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
            x = sorted(years_data.keys())
            y = [years_data[year][i] for year in x]
            data.append(go.Bar(name=sentiment, x=x, y=y, visible=False if movie != all_movies[0] else True))

    layout = go.Layout(
        title=f"Sentiment Analysis for {all_movies[0]}",
        xaxis={"title": "Year"},
        yaxis={"title": "Review Count"},
        barmode='stack',
        updatemenus=[{"buttons": dropdown_options, "direction": "down", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top"}],
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='Page_and_JSON/6.7-plot_TwoBar.html', auto_open=False)

if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'
    data = load_data(filepath)
    sentiment_counts = preprocess_data(data)
    create_visualization(sentiment_counts)

