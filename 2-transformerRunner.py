import torch
import transformers
from torch import nn
import pandas as pd
import re
from html import unescape
# 定义Transformer模型类
class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze=True):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        output = self.transformer(ids, output_attentions=False)
        hidden = output.last_hidden_state
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        return prediction

# 加载tokenizer和模型
transformer_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)
transformer_model = transformers.AutoModel.from_pretrained(transformer_name)

# 定义模型
output_dim = 2  # 假设有2个输出类别，例如正面和负面情感
model = Transformer(transformer_model, output_dim, freeze=True)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载训练好的模型权重
model.load_state_dict(torch.load("transformerThreeRecallTest.pt", map_location=device))
print("模型加载完毕")
# 定义预测函数
def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    # 使用truncation和max_length参数来自动截断超长序列
    ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(ids)
        predictions = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        predicted_probability = predictions[0][predicted_class].item()
    return predicted_class, predicted_probability
def clean_text(text):
    # HTML解码
    text = unescape(text)
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 移除电子邮件地址
    text = re.sub(r'\S*@\S*\s?', '', text)
    # 替换数字为特定标记（或可以选择移除它们）
    text = re.sub(r'\d+', ' ', text)
    # 移除特殊符号和标点（根据需要调整）
    text = re.sub(r'[^\w\s]', '', text)
    # 替换连续空白字符为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除文本开头和结尾的空白字符
    text = text.strip()
    return text
# 使用模型进行预测
# text = "This is one bad movie!"
# predicted_class, predicted_probability = predict_sentiment(text, model, tokenizer, device)
# print(f"Predicted class: {predicted_class}, Probability: {predicted_probability:.4f}")
# 设定文件名
# file_name = 'A-IMDB_Reviews.xlsx'
# file_name = 'A-IMDB_Reviews.xlsx'
file_name = 'Excel_File/A-IMDB_Reviews.xlsx'
# 读取xls文件
df = pd.read_excel(file_name)

# 确定'Review'列的位置
review_col_index = df.columns.get_loc("Review")
# 初始化一个列表来收集字典
cleaned_data = []
# 在'Review'列前插入新列以存储预测结果和概率
df.insert(review_col_index, 'Probability', pd.NA)
df.insert(review_col_index, 'Predicted Class', pd.NA)

print("进入影评预测")
# 遍历影评进行预测，并将预测结果填入新列
for index, row in df.iterrows():
    # text = row['Review']
    clean_review = clean_text(row['Review'])  # 清洗文本
    predicted_class, predicted_probability = predict_sentiment(clean_review, model, tokenizer, device)
    df.at[index, 'Predicted Class'] = predicted_class
    df.at[index, 'Probability'] = predicted_probability
    # 收集清洗后 DataFrame 的数据
    cleaned_data.append(
        {'Cleaned Review': clean_review, 'Predicted Class': predicted_class, 'Probability': predicted_probability})

# Create a new DataFrame from the list of dictionaries
df_cleaned = pd.DataFrame(cleaned_data)
# 保存修改后的文件
# modified_file_name = 'BThree-Modified_IMDB_Reviews.xlsx'
modified_file_name = 'Excel_File/BThree-Modified_IMDB_Reviews.xlsx'
df.to_excel(modified_file_name, index=False)
# 注意：根据你的具体任务，你可能需要调整output_dim和类别映射（如何从predicted_class到实际的情感标签）。
# 保存只包含清洗后评论和预测结果的文件
# 从字典列表创建一个新的 DataFrame
# cleaned_file_name = 'B2Three-Cleaned_IMDB_Reviews_Predictions.xlsx'
cleaned_file_name = 'Excel_File/b2Three-Cleaned_IMDB_Reviews_Predictions.xlsx'
df_cleaned.to_excel(cleaned_file_name, index=False)