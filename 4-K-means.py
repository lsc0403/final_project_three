# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.cluster import KMeans
#
# # 加载数据
# file_path = 'Excel_File/K-means_ready_for_generation.xlsx'
# data = pd.read_excel(file_path)
#
# # 初始化标签编码器
# le_up_vote = LabelEncoder()
# le_total_vote = LabelEncoder()
# le_rating = LabelEncoder()
# le_predicted_class = LabelEncoder()
# le_review = LabelEncoder()
#
# # 对分类变量进行编码
# data['Up Vote Cat Encoded'] = le_up_vote.fit_transform(data['Up Vote Cat'])
# data['Total Vote Cat Encoded'] = le_total_vote.fit_transform(data['Total Vote Cat'])
# data['Rating Cat Encoded'] = le_rating.fit_transform(data['Rating Cat'])
# data['Predicted Class Cat Encoded'] = le_predicted_class.fit_transform(data['Predicted Class Cat'])
# data['Review Cat Encoded'] = le_review.fit_transform(data['Review Cat'])
# # 选择用于聚类的特征
# features = data[['Up Vote Cat Encoded', 'Total Vote Cat Encoded', 'Rating Cat Encoded',
#                  'Predicted Class Cat Encoded', 'Review Cat Encoded']]
#
# # 执行K-means聚类
# kmeans = KMeans(n_clusters=6, random_state=42)
# data['Cluster'] = kmeans.fit_predict(features)
#
# # 查看聚类结果
# print(data[['Up Vote Cat', 'Total Vote Cat', 'Rating Cat', 'Predicted Class Cat', 'Review Cat', 'Cluster']].head())
# # 分析每个聚类中的分布
# for cluster in range(kmeans.n_clusters): # 假设我们有4个聚类
#     print(f"Cluster {cluster} distribution:")
#     print(data[data['Cluster'] == cluster]['Predicted Class Cat'].value_counts())
#     print("\n")
#     print(f"Cluster {cluster} summary:")
#
#     # 计算并输出原始分类特征的最频繁类别
#     for column in ['Up Vote Cat', 'Total Vote Cat', 'Rating Cat', 'Predicted Class Cat', 'Review Cat']:
#         most_frequent = data[data['Cluster'] == cluster][column].mode()[0]
#         print(f"Most frequent {column}: {most_frequent}")
#
#     # 输出编码后的特征的平均值
#     for column in ['Up Vote Cat Encoded', 'Total Vote Cat Encoded', 'Rating Cat Encoded',
#                    'Predicted Class Cat Encoded', 'Review Cat Encoded']:
#         mean_value = data[data['Cluster'] == cluster][column].mean()
#         print(f"{column} mean: {mean_value}")
#
#     print("\n")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# 加载数据
file_path = 'Excel_File/K-means_ready_for_generation.xlsx'
data = pd.read_excel(file_path)

# 初始化标签编码器
le_up_vote = LabelEncoder()
le_total_vote = LabelEncoder()
# le_rating = LabelEncoder()
le_predicted_class = LabelEncoder()
le_review = LabelEncoder()

# 对分类变量进行编码
data['Up Vote Cat Encoded'] = le_up_vote.fit_transform(data['Up Vote Cat'])
data['Total Vote Cat Encoded'] = le_total_vote.fit_transform(data['Total Vote Cat'])
# data['Rating Cat Encoded'] = le_rating.fit_transform(data['Rating Cat'])
data['Predicted Class Cat Encoded'] = le_predicted_class.fit_transform(data['Predicted Class Cat'])
data['Review Cat Encoded'] = le_review.fit_transform(data['Review Cat'])
# 选择用于聚类的特征
features = data[['Up Vote Cat Encoded', 'Total Vote Cat Encoded', 'Review Cat Encoded']]

# 执行K-means聚类
kmeans = KMeans(n_clusters=8, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# 查看聚类结果
print(data[['Up Vote Cat', 'Total Vote Cat', 'Review Cat', 'Cluster']].head())
# 分析每个聚类中的分布
for cluster in range(kmeans.n_clusters): # 假设我们有4个聚类
    print(f"Cluster {cluster} distribution:")
    print(data[data['Cluster'] == cluster]['Predicted Class Cat'].value_counts())
    print("\n")
    print(f"Cluster {cluster} summary:")

    # 计算并输出原始分类特征的最频繁类别
    for column in ['Up Vote Cat', 'Total Vote Cat', 'Review Cat']:
        most_frequent = data[data['Cluster'] == cluster][column].mode()[0]
        print(f"Most frequent {column}: {most_frequent}")

    # 输出编码后的特征的平均值
    for column in ['Up Vote Cat Encoded', 'Total Vote Cat Encoded', 'Review Cat Encoded']:
        mean_value = data[data['Cluster'] == cluster][column].mean()
        print(f"{column} mean: {mean_value}")

    print("\n")