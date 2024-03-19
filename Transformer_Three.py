import pandas as pd
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transformers
import time
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果的可重复性
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 读取Excel文件并转换为datasets.Dataset格式
df = pd.read_excel("ReviewTotal140k.xlsx")
df['label'] = df['label'].map({'Positive': 0, 'Neutral': 1, 'Negative': 2})

# 将pandas DataFrame分割为训练集、验证集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)

# 使用datasets.Dataset.from_pandas转换DataFrame
train_data = Dataset.from_pandas(train_df)
valid_data = Dataset.from_pandas(valid_df)
test_data = Dataset.from_pandas(test_df)

# 初始化transformer模型的tokenizer
transformer_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)

# 定义一个函数用于文本的tokenize和numericalize
def tokenize_and_numericalize_example(example):
    output = tokenizer(example["ids"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {"ids": output["input_ids"].squeeze(0), "attention_mask": output["attention_mask"].squeeze(0)}

# 对数据集应用上述函数
train_data = train_data.map(tokenize_and_numericalize_example)
valid_data = valid_data.map(tokenize_and_numericalize_example)
test_data = test_data.map(tokenize_and_numericalize_example)

# 设置数据格式为torch，选定需要的列
train_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
valid_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])

pad_index = tokenizer.pad_token_id

# 定义批量数据处理函数
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_masks = [i["attention_mask"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
        batch_masks = nn.utils.rnn.pad_sequence(batch_masks, padding_value=0, batch_first=True)
        batch_label = torch.tensor([i["label"] for i in batch])
        batch = {"ids": batch_ids, "attention_mask": batch_masks, "label": batch_label}
        return batch
    return collate_fn

# 定义数据加载器
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return data_loader

# 设置batch大小并获取数据加载器
batch_size = 8
train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

# 定义transformer模型
class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids, attention_mask=None):
        output = self.transformer(ids, attention_mask=attention_mask, return_dict=True)
        hidden = output.last_hidden_state
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(cls_hidden)
        return prediction

# 从预训练模型加载transformer
transformer = transformers.AutoModel.from_pretrained(transformer_name)
transformer.config.hidden_size

# 定义模型输出维度和是否冻结transformer层的参数
output_dim = len(train_data["label"].unique())
freeze = False
model = Transformer(transformer, output_dim, freeze)

# 计算模型的可训练参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(model):,} trainable parameters")

# 设置优化器
lr = 1e-5   #原始为lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = model.to(device)
criterion = criterion.to(device)

# 定义训练和评估函数
def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  # 新增行
        label = batch["label"].to(device)
        prediction = model(ids, attention_mask=attention_mask)  # 修改这行
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # 新增行
            label = batch["label"].to(device)
            prediction = model(ids, attention_mask=attention_mask)  # 修改这行
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

# 定义准确率计算函数
def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

# 运行训练和验证循环
n_epochs = 4
# # 引入学习率调度器 ********************
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=n_epochs, steps_per_epoch=len(train_data_loader))

best_valid_loss = float("inf")
metrics = collections.defaultdict(list)
for epoch in range(n_epochs):
    train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, device)
    # scheduler.step()  # 在每个epoch后更新学习率  ******************************
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "transformerThreeAllData.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
    # 暂停半小时
    if epoch < n_epochs - 1:  # 如果不是最后一个epoch，才暂停
        print("Pausing for 30 minutes...")
        time.sleep(1800)  # 暂停1800秒（即半小时
# 绘制训练和验证的损失曲线
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()

# 绘制训练和验证的准确率曲线
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()

# 加载最佳模型并在测试集上评估
model.load_state_dict(torch.load("transformerThreeAllData.pt"))
test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

# 定义情感预测函数
def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

# 使用情感预测函数进行预测
text = "This film is terrible!"
predict_sentiment(text, model, tokenizer, device)
text = "This film is great!"
predict_sentiment(text, model, tokenizer, device)
text = "This film is not terrible, it's great!"
predict_sentiment(text, model, tokenizer, device)
text = "This film is not great, it's terrible!"
predict_sentiment(text, model, tokenizer, device)
