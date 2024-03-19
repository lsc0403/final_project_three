import json
import plotly.graph_objs as go
from plotly.offline import plot

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    # 汇总情感分类统计信息
    sentiment_counts = {}
    for movie, reviews in data.items():
        counts = {0: 0, 1: 0, 2: 0}  # 正向，中性，负向
        for review in reviews:
            counts[review['Predicted Class']] += 1
        sentiment_counts[movie] = counts
    return sentiment_counts

def create_visualization(sentiment_counts):
    traces = []
    for movie in sentiment_counts:
        counts = sentiment_counts[movie]
        total_reviews = sum(counts.values())
        positive_percentage = counts[0] / total_reviews * 100 if total_reviews > 0 else 0
        movie_label = f"{movie} (Reviews: {total_reviews}, Positive: {positive_percentage:.2f}%)"

        values = [counts[i] / total_reviews * 100 for i in range(3)] if total_reviews > 0 else [0, 0, 0]

        trace = go.Bar(
            x=[movie]*3,
            y=values,
            name=movie_label,
            hoverinfo='y+name',
            text=[f"{counts[i]} reviews" for i in range(3)],
            textposition='auto',
            marker=dict(
                line=dict(
                    color='rgba(0, 0, 0, 1)',
                    width=2
                )
            ),
            visible='legendonly'
        )
        traces.append(trace)

    layout = go.Layout(
        title={
            'text': 'Movie Reviews Sentiment Analysis',
            'y': 0.9,
            'x': 0.2,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(title='Movie'),
        yaxis=dict(title='Percentage'),
        barmode='stack',
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'method': 'restyle',
                'args': [{'visible': [True]*len(traces)}],
                'label': 'Show All',
            }],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.2,
            'xanchor': 'center',
            'y': 1.2,
            'yanchor': 'top'
        }],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=100)
    )

    fig = go.Figure(data=traces, layout=layout)
    plot(fig, filename='Page_and_JSON/6.6-plot_bar.html', auto_open=False)

if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'
    data = load_data(filepath)
    sentiment_counts = preprocess_data(data)
    create_visualization(sentiment_counts)




