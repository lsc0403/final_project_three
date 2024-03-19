import json
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os


# 定义函数读取数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


# 生成词云图并保存为图片
def generate_wordclouds(data, output_dir):
    for movie, reviews in data.items():
        # 汇总所有影评文本，并移除指定词语
        # all_reviews = " ".join([review["Review"].replace("film", "").replace("movie", "").replace("one", "") for review in reviews])
        all_reviews = " ".join(
            [str(review["Review"]).replace("film", "").replace("movie", "").replace("one", "") for review in reviews])

        # 创建词云对象
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
        # 保存词云图为图片
        output_path = os.path.join(output_dir, f"{movie}_wordcloud.png")
        wordcloud.to_file(output_path)


# 生成HTML文件展示词云图
def generate_html(data, output_dir,output_total):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movie Reviews Wordclouds</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .title {
                text-align: center;
                margin-bottom: 20px;
                font-size: 24px;
            }
            .btn {
                display: block;
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                font-size: 16px;
                background-color: #007bff;
                color: #fff;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            .wordcloud-img {
                display: none;
                max-width: 100%;
                height: auto;
                margin-top: 10px;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
            }
        </style>
        <script>
        function showWordcloud(movie) {
            var image = document.getElementById(movie);
            if (image.style.display === 'none') {
                image.style.display = 'block';
            } else {
                image.style.display = 'none';
            }
        }
        </script>
    </head>
    <body>
    <div class="container">
        <h2 class="title">Word Clouds</h2>
    """
    for movie, _ in data.items():
        # 处理电影名中的引号
        movie_formatted = movie.replace("'", "\\'")
        image_path = f"{output_dir}/{movie}_wordcloud.png"  # 使用相对路径
        html_content += f"<button class='btn' onclick='showWordcloud(\"{movie_formatted}\")'>{movie}</button>"
        html_content += f"<img id='{movie_formatted}' class='wordcloud-img' src='{image_path}'>"
    html_content += """
    </div>
    </body>
    </html>
    """

    with open("Page_and_JSON/6.4-plot_cloud.html", "w") as html_file:  # 输出到代码同一文件夹下
        html_file.write(html_content)


if __name__ == "__main__":
    filepath = 'Page_and_JSON/BThree-JSONData.json'  # 更新为实际的文件路径
    output_dir = 'wordcloud_images'  # 输出词云图的文件夹
    output_dir2 = 'Page_and_JSON'
    output_total = os.path.join(output_dir2, output_dir)
    os.makedirs(output_total, exist_ok=True)

    data = load_data(filepath)
    generate_wordclouds(data, output_total)
    generate_html(data, output_dir,output_total)
