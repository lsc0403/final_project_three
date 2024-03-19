# import json
# from collections import defaultdict
# import plotly.graph_objs as go
# from plotly.offline import plot
# from plotly.subplots import make_subplots
#
# def load_data(filepath):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data
#
# def preprocess_data(data):
#     sentiment_counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # movie -> year -> [positive, neutral, negative]
#     for movie, reviews in data.items():
#         for review in reviews:
#             year = review['Date'].split("-")[0]  # Assuming date format is YYYY-MM-DD
#             sentiment = review['Predicted Class']
#             sentiment_counts[movie][year][sentiment] += 1
#     return sentiment_counts
#
# def create_visualization(sentiment_counts):
#     all_movies = list(sentiment_counts.keys())
#     dropdown_options = []
#     fig = make_subplots(rows=1, cols=2, subplot_titles=("Count-based Stacked Bars", "Proportion-based Stacked Bars"))
#
#     for movie, years_data in sentiment_counts.items():
#         years = sorted(years_data.keys())
#         counts = [[years_data[year][i] for year in years] for i in range(3)]  # Count-based data
#         totals = [sum(years_data[year]) for year in years]
#         proportions = [[count / total if total else 0 for count, total in zip(counts[i], totals)] for i in range(3)]  # Proportion-based data
#
#         for i, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
#             fig.add_trace(go.Bar(name=sentiment, x=years, y=counts[i], visible=movie == all_movies[0]), row=1, col=1)
#             fig.add_trace(go.Bar(name=sentiment, x=years, y=proportions[i], visible=movie == all_movies[0]), row=1, col=2)
#
#         dropdown_options.append({"label": movie, "method": "update", "args": [{"visible": [movie == m for m in all_movies for _ in range(6)]}, {"title": f"Sentiment Analysis for {movie}"}]})
#
#     fig.update_layout(
#         barmode='stack',
#         updatemenus=[{"buttons": dropdown_options, "direction": "down", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top"}],
#         title=f"Sentiment Analysis for {all_movies[0]}"
#     )
#
#     plot(fig, filename='sentiment_analysis_combined.html', auto_open=True)
#
# if __name__ == "__main__":
#     filepath = 'BThree-JSONData.json'
#     data = load_data(filepath)
#     sentiment_counts = preprocess_data(data)
#     create_visualization(sentiment_counts)
import json
from collections import defaultdict
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    sentiment_counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # movie -> year -> [positive, neutral, negative]
    for movie, reviews in data.items():
        for review in reviews:
            year = review['Date'].split("-")[0]  # Assuming date format is YYYY-MM-DD
            sentiment = review['Predicted Class']
            sentiment_counts[movie][year][sentiment] += 1
    return sentiment_counts

def create_visualization(sentiment_counts):
    all_movies = list(sentiment_counts.keys())
    dropdown_options = []
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Count-based Stacked Bars", "Proportion-based Stacked Bars"))

    for movie in all_movies:
        years_data = sentiment_counts[movie]
        years = sorted(years_data.keys())
        counts = [[years_data[year][i] for year in years] for i in range(3)]  # Count-based data
        totals = [sum(years_data[year]) for year in years]
        total_reviews = sum(totals)
        positive_proportion = sum(counts[0]) / total_reviews if total_reviews else 0
        proportions = [[count / total if total else 0 for count, total in zip(counts[i], totals)] for i in range(3)]  # Proportion-based data

        for i, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
            fig.add_trace(go.Bar(name=sentiment, x=years, y=counts[i], visible=movie == all_movies[0]), row=1, col=1)
            fig.add_trace(go.Bar(name=sentiment, x=years, y=proportions[i], visible=movie == all_movies[0]), row=1, col=2)

        dropdown_label = f"{movie} (Total Reviews: {total_reviews}, Positive: {positive_proportion:.2%})"
        dropdown_options.append({"label": dropdown_label, "method": "update", "args": [{"visible": [movie == m for m in all_movies for _ in range(6)]}, {"title": f"Sentiment Analysis for {movie}"}]})

    fig.update_layout(
        barmode='stack',
        updatemenus=[{"buttons": dropdown_options, "direction": "down", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.3, "xanchor": "left", "y": 1.15, "yanchor": "top"}],
        title=f"Sentiment Analysis for {all_movies[0]}"
    )

    plot(fig, filename='Page_and_JSON/6.9-plot_bar_merger.html', auto_open=False)

if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'
    data = load_data(filepath)
    sentiment_counts = preprocess_data(data)
    create_visualization(sentiment_counts)
