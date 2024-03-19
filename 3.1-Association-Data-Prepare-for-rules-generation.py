import pandas as pd

def transform_reviews_data(file_path):
    # 加载数据
    data = pd.read_excel(file_path)
    # Calculate the average length of reviews
    average_length = data['Review'].apply(lambda x: len(x.split())).mean()

    # Define thresholds for review length categories
    # For simplicity, you might consider "short" as below average, "medium" as around average,
    # and "long" as above average. Adjust these definitions as needed.
    short_threshold = average_length * 0.5  # Reviews with less than 50% of the average length
    long_threshold = average_length * 1.5  # Reviews with more than 150% of the average length

    # Categorize review lengths
    data['Review Cat'] = data['Review'].apply(lambda x: 'long' if len(x.split()) > long_threshold
    else 'medium' if len(x.split()) > short_threshold
    else 'short')
    # 转换数据
    data['Up Vote Cat'] = ['High' if x > data['Up Vote'].median() else 'Low' for x in data['Up Vote']]
    data['Total Vote Cat'] = ['High' if x > data['Total Vote'].median() else 'Low' for x in data['Total Vote']]
    data['Rating Cat'] = pd.cut(data['Rating'], bins=[0, 3, 6, 10], labels=['Low', 'Medium', 'High'], right=True)
    # data['Probability Cat'] = ['High' if x >= 0.55 else 'Low' for x in data['Probability']]
    # data['Predicted Class Cat'] = ['Positive' if x == 1 else 'Negative' for x in data['Predicted Class']]
    # data['Predicted Class Cat'] = ['Positive' if x == 0 else 'Neutral' if x == 1 else 'Negative' if x == 2 else 'Wrong'
    #                                for x in data['Predicted Class']]
    # 选择转换后的列
    transformed_data = data[['Movie Name', 'Title', 'Author', 'Date', 'Review',
                             'Up Vote Cat', 'Total Vote Cat', 'Rating Cat',
                             'Predicted Class Cat']]

    # 保存到新的Excel文件
    # output_file_path = 'CThree-Transformed_IMDB_Reviews.xlsx'
    output_file_path = 'Excel_File/K-means_ready_for_generation.xlsx'
    transformed_data.to_excel(output_file_path, index=False)

    print(f'Transformed data saved to {output_file_path}')

# 执行转换
# file_path = 'BThree-Modified_IMDB_Reviews.xlsx'
file_path = 'Excel_File/K-means_rules_dataset.xlsx'
transform_reviews_data(file_path)
