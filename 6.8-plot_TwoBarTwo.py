import json
from collections import defaultdict
import plotly.graph_objs as go
from plotly.offline import plot

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    sentiment_distribution = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # 电影 -> 年份 -> [正向, 中性, 负向]
    for movie, reviews in data.items():
        for review in reviews:
            year = review['Date'].split("-")[0]
            predicted_class = review['Predicted Class']
            sentiment_distribution[movie][year][predicted_class] += 1
    return sentiment_distribution

def create_html(sentiment_distribution):
    traces = []
    dropdown_options = []
    for movie, yearly_data in sentiment_distribution.items():
        years = []
        proportions = [[], [], []]  # 正面、中性、负面
        for year, sentiments in sorted(yearly_data.items()):
            total_reviews = sum(sentiments)
            if total_reviews == 0: continue
            for i in range(3):
                proportions[i].append(sentiments[i] / total_reviews)
            years.append(year)
        for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
            traces.append(go.Bar(x=years, y=proportions[i], name=sentiment, visible=True if movie == list(sentiment_distribution.keys())[0] else False))
        dropdown_options.append({
            'label': movie,
            'method': 'update',
            'args': [{'visible': [movie == m for m in sentiment_distribution.keys() for _ in range(3)]}]
        })

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='stack',
        updatemenus=[{
            'buttons': dropdown_options,
            'direction': 'down',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        title='Sentiment Analysis by Year'
    )

    plot(fig, filename='Page_and_JSON/6.8-plot_TwoBarTwo.html', auto_open=False)

if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'
    data = load_data(filepath)
    sentiment_distribution = preprocess_data(data)
    create_html(sentiment_distribution)


