import pandas as pd

# 读取Excel文件
df = pd.read_excel('Excel_File/positive_for_use.xlsx')

# 按照'Movie Name'分组，然后每组只保留前70行
result_df = df.groupby('Movie Name').head(600)

# 将处理后的数据写回到一个新的Excel文件
result_df.to_excel('filtered_test.xlsx', index=False)
