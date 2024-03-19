import json
from collections import defaultdict, Counter
from datetime import datetime

# 读取JSON数据的函数
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# 分析电影评价数据的函数
def analyze_reviews(data):
    results = {}
    for movie, reviews in data.items():
        total_ratings = sum(review['Rating'] for review in reviews)
        total_reviews = len(reviews)
        # 保留两位小数
        average_rating = round(total_ratings / total_reviews, 2) if total_reviews else 0
        ratings_counter = Counter(review['Rating'] for review in reviews)
        most_common_rating = ratings_counter.most_common(1)[0][0] if ratings_counter else 0
        positive_reviews_percentage = sum(1 for review in reviews if review['Rating'] >= 4) / total_reviews * 100
        average_review_length = sum(
            len(str(review['Review']).split()) for review in reviews) / total_reviews if total_reviews else 0

        # 计算收获最多评价的年份
        reviews_by_year = defaultdict(int)
        for review in reviews:
            year = datetime.strptime(review['Date'], '%Y-%m-%d').year
            reviews_by_year[year] += 1
        most_reviews_year = max(reviews_by_year, key=reviews_by_year.get) if reviews_by_year else None

        # 确定电影评价质量
        if positive_reviews_percentage >= 95:
            review_quality = "praised overwhelmingly"
        elif positive_reviews_percentage >= 80:
            review_quality = "highly praised"
        elif positive_reviews_percentage >= 70:
            review_quality = "mostly positive reviews"
        elif positive_reviews_percentage > 40:
            review_quality = "mixed reviews"
        elif positive_reviews_percentage > 30:
            review_quality = "mostly negative reviews"
        elif positive_reviews_percentage > 20:
            review_quality = "highly criticized"
        else:
            review_quality = "criticized overwhelmingly"

        results[movie] = {
            "review_quality": review_quality,
            "total_reviews": total_reviews,
            "average_rating": average_rating,
            "most_common_rating": most_common_rating,
            "positive_reviews_percentage": positive_reviews_percentage,
            "average_review_length": average_review_length,
            "most_reviews_year": most_reviews_year
        }
    return results

# 生成HTML文件的函数，包含页面排版改进
def generate_html_fixed(descriptions, filename="movie_reviews_final.html"):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Reviews Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f0f0f0; }}
            .movie {{ margin-bottom: 20px; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #007bff; color: white; }}
            .title, .description {{ font-weight: bold; cursor: pointer; }}
            .description {{ margin-top: 5px; display: none; color: black; }} /* 修改字体颜色为黑色并加粗 */
            h1 {{ color: #444; }}
        </style>
        <script>
            function toggleDescription(movieId) {{
                var description = document.getElementById(movieId);
                if (description.style.display === "none") {{
                    description.style.display = "block";
                }} else {{
                    description.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <h1>Movie Reviews Analysis</h1>
        {movies_html}
    </body>
    </html>
    """

    movies_html = ""
    for idx, (movie, desc) in enumerate(descriptions.items()):
        movie_id = f"description_{idx}"  # Unique ID for each description
        movies_html += f"""<div class='movie'>
                            <div class='title' onclick='toggleDescription("{movie_id}")'>{movie}</div>
                            <div id='{movie_id}' class='description' style='color: black; font-weight: bold;'>{desc}</div>
                           </div>\n"""

    html_content = html_template.format(movies_html=movies_html)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(html_content)

    return filename




# 主程序
if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'  # 确保文件路径正确
    # filepath = 'F-JSONData.json'  # 确保文件路径正确
    data = load_data(filepath)
    movie_reviews_analysis = analyze_reviews(data)

    # 根据分析结果生成英文描述
    descriptions = {}
    for movie, info in movie_reviews_analysis.items():
        desc = (f"This is a {info['review_quality']} movie, with {info['total_reviews']} reviews collected, "
                f"the average rating is {info['average_rating']:.2f} stars, the most common rating is {info['most_common_rating']} stars, "
                f"with {info['positive_reviews_percentage']:.2f}% of people giving the movie a positive review, "
                f"the average review length is {info['average_review_length']:.2f} words, "
                f"and the year {info['most_reviews_year']} saw the most reviews for this movie.")
        descriptions[movie] = desc

    # 生成并保存HTML文件
    html_filename = generate_html_fixed(descriptions, "Page_and_JSON/7-ReviewSummaryGeneration.html")
    print(f"HTML file generated: {html_filename}")
