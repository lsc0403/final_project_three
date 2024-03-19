import pandas as pd

# 加载数据
data = pd.read_excel('Excel_File/BThree-Modified_IMDB_Reviews.xlsx')

# 计算Up Vote和Total Vote的中位数
up_vote_median = data['Up Vote'].median()
total_vote_median = data['Total Vote'].median()

# 复制一份数据用于比较
data_before = data.copy()

# 应用条件更新Predicted Class列
#cluster=6
data.loc[
    (data['Probability'] < 0.8) &
    (data['Up Vote'] > up_vote_median) &
    (data['Total Vote'] > total_vote_median) &
    (data['Rating'].between(0, 3, inclusive="both")), 'Predicted Class'] = 0
#cluster=6
data.loc[
    (data['Probability'] < 0.8) &
    (data['Up Vote'] > up_vote_median) &
    (data['Total Vote'] > total_vote_median) &
    (data['Rating'].between(6, 10, inclusive="both")), 'Predicted Class'] = 1
#cluster=3
data.loc[
    (data['Probability'] < 0.8) &
    (data['Up Vote'] < up_vote_median) &
    (data['Total Vote'] < total_vote_median) &
    (data['Rating'].between(6, 10, inclusive="both")), 'Predicted Class'] = 1
#cluster=6
data.loc[
    (data['Probability'] < 0.8) &
    (data['Up Vote'] > up_vote_median) &
    (data['Total Vote'] < total_vote_median) &
    (data['Rating'].between(6, 10, inclusive="both")), 'Predicted Class'] = 1
# 比较更改前后的Predicted Class列，找出被修改的行
modified_rows = data[data['Predicted Class'] != data_before['Predicted Class']]

# 保存更新后的DataFrame
data.to_excel('Excel_File/DThree-Modified_IMDB_Reviews_Updated.xlsx', index=False)

# 输出被修改的行
print("被修改的行：")
print(modified_rows)

