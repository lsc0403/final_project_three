# import pandas as pd
# import collections
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import tqdm
# import transformers
# import time
# from datasets import Dataset, DatasetDict
# from sklearn.model_selection import train_test_split
# # Set random seed for reproducibility
# seed = 1234
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
#
# # Read Excel file and convert to datasets.Dataset format
# df = pd.read_excel("ReviewSmallSizeForTest1k.xlsx")
# df['label'] = df['label'].map({'Positive': 0, 'Neutral': 1, 'Negative': 2})
#
# # Split pandas DataFrame into train, validation, and test sets
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
# valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)
#
# # Convert DataFrame to Dataset
# train_data = Dataset.from_pandas(train_df)
# valid_data = Dataset.from_pandas(valid_df)
# test_data = Dataset.from_pandas(test_df)
#
# # Initialize transformer model's tokenizer
# transformer_name = "bert-base-uncased"
# tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)
#
#
# # Define a function for tokenization and numericalization
# def tokenize_and_numericalize_example(example):
#     output = tokenizer(example["ids"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
#     return {"ids": output["input_ids"].squeeze(0), "attention_mask": output["attention_mask"].squeeze(0)}
#
#
# # Apply the function to the dataset
# train_data = train_data.map(tokenize_and_numericalize_example)
# valid_data = valid_data.map(tokenize_and_numericalize_example)
# test_data = test_data.map(tokenize_and_numericalize_example)
#
# # Set the data format to torch and select the needed columns
# train_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
# valid_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
# test_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
#
# pad_index = tokenizer.pad_token_id
#
#
# # Define batch data processing function
# def get_collate_fn(pad_index):
#     def collate_fn(batch):
#         batch_ids = [i["ids"] for i in batch]
#         batch_masks = [i["attention_mask"] for i in batch]
#         batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
#         batch_masks = nn.utils.rnn.pad_sequence(batch_masks, padding_value=0, batch_first=True)
#         batch_label = torch.tensor([i["label"] for i in batch])
#         batch = {"ids": batch_ids, "attention_mask": batch_masks, "label": batch_label}
#         return batch
#
#     return collate_fn
#
#
# # Define data loader
# def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
#     collate_fn = get_collate_fn(pad_index)
#     data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn,
#                                               shuffle=shuffle)
#     return data_loader
#
#
# # Set batch size and get data loaders
# batch_size = 8
# train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
# valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
# test_data_loader = get_data_loader(test_data, batch_size, pad_index)
#
#
# # Define transformer model
# class Transformer(nn.Module):
#     def __init__(self, transformer, output_dim, freeze):
#         super().__init__()
#         self.transformer = transformer
#         hidden_dim = transformer.config.hidden_size
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         if freeze:
#             for param in self.transformer.parameters():
#                 param.requires_grad = False
#
#     def forward(self, ids, attention_mask=None):
#         output = self.transformer(ids, attention_mask=attention_mask, return_dict=True)
#         hidden = output.last_hidden_state
#         cls_hidden = hidden[:, 0, :]
#         prediction = self.fc(cls_hidden)
#         return prediction
#
#
# # Load transformer from pre-trained model
# transformer = transformers.AutoModel.from_pretrained(transformer_name)
# output_dim = len(train_data["label"].unique())  # Number of unique labels
# freeze = False
# model = Transformer(transformer, output_dim, freeze)
#
#
# # Count model's trainable parameters
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# print(f"The model has {count_parameters(model):,} trainable parameters")
#
# # Set optimizer
# lr = 1e-5
# optimizer = optim.Adam(model.parameters(), lr=lr)
#
# # Define loss function
# criterion = nn.CrossEntropyLoss()
#
# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# criterion = criterion.to(device)
#
#
# # Update get_metrics function to calculate accuracy and recall
# def get_metrics(prediction, label, num_classes):
#     predicted_classes = prediction.argmax(dim=-1)
#     correct_predictions = predicted_classes.eq(label).sum()
#     accuracy = correct_predictions / label.shape[0]
#
#     # Initialize containers for true positives and false negatives
#     true_positives = torch.zeros(num_classes)
#     false_negatives = torch.zeros(num_classes)
#
#     for class_id in range(num_classes):
#         true_positives[class_id] = ((predicted_classes == class_id) & (label == class_id)).sum()
#         false_negatives[class_id] = ((predicted_classes != class_id) & (label == class_id)).sum()
#
#     # Calculate recall for each class to avoid division by zero, add a small epsilon
#     recalls = (true_positives / (true_positives + false_negatives + 1e-6)).tolist()
#
#     return accuracy.item(), recalls
#
#
# # Define train function
# def train(data_loader, model, criterion, optimizer, device, num_classes):
#     model.train()
#     epoch_losses = []
#     epoch_accs = []
#     epoch_recalls = [[] for _ in range(num_classes)]  # List of lists to hold recalls for each class
#     for batch in tqdm.tqdm(data_loader, desc="Training"):
#         ids = batch["ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         label = batch["label"].to(device)
#         prediction = model(ids, attention_mask=attention_mask)
#         loss = criterion(prediction, label)
#         accuracy, recalls = get_metrics(prediction, label, num_classes)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_losses.append(loss.item())
#         epoch_accs.append(accuracy)
#         for i, recall in enumerate(recalls):
#             epoch_recalls[i].append(recall)
#     # Calculate the average recall for each class
#     average_recalls = [np.mean(recall) for recall in epoch_recalls]
#     return np.mean(epoch_losses), np.mean(epoch_accs), average_recalls
#
#
#
# # Define evaluate function
# def evaluate(data_loader, model, criterion, device, num_classes):
#     model.eval()
#     epoch_loss, epoch_accuracy = 0, 0
#     recalls = {class_id: [] for class_id in range(num_classes)}
#     with torch.no_grad():
#         for batch in data_loader:
#             ids = batch['ids'].to(device)  # 修改这里
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
#
#             predictions = model(ids, attention_mask)  # 注意这里也使用了 'ids'
#             loss = criterion(predictions, labels)
#             accuracy, recall_per_class = get_metrics(predictions, labels, num_classes)
#
#             epoch_loss += loss.item()
#             epoch_accuracy += accuracy
#             for class_id, recall in enumerate(recall_per_class):
#                 recalls[class_id].append(recall)
#
#     return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader), {class_id: np.mean(recalls[class_id]) for class_id in range(num_classes)}
#
#
#
# # Training and evaluation loop
# n_epochs = 3
# best_valid_loss = float("inf")
#
# metrics = {"train_losses": [], "train_accs": [], "train_recalls": [], "valid_losses": [], "valid_accs": [],
#            "valid_recalls": []}
# # 为每个类动态添加召回率的键。
# for i in range(output_dim):
#     metrics[f"train_recall_class_{i}"] = []
#     metrics[f"valid_recall_class_{i}"] = []
# for epoch in range(n_epochs):
#     train_loss, train_acc, train_recalls = train(train_data_loader, model, criterion, optimizer, device, output_dim)
#     valid_loss, valid_acc, valid_recalls = evaluate(valid_data_loader, model, criterion, device, output_dim)
#
#     metrics["train_losses"].append(train_loss)
#     metrics["train_accs"].append(train_acc)
#     for i, recall in enumerate(train_recalls):
#         metrics[f"train_recall_class_{i}"].append(recall)
#
#     metrics["valid_losses"].append(valid_loss)
#     metrics["valid_accs"].append(valid_acc)
#     for i, recall in enumerate(valid_recalls):
#         metrics[f"valid_recall_class_{i}"].append(recall)
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), "transformerThreeRecallTest.pt")
#     print(
#         f"\nEpoch: {epoch}:\n Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train Recalls: {[round(recall, 3) for recall in train_recalls]}, \nValid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid Recalls: {[round(recall, 3) for recall in valid_recalls.values()]}")
#
#     # The remaining parts of your original code related to plotting and testing can be included here without modification.
#
#     # 暂停半小时
#     # if epoch < n_epochs - 1:  # 如果不是最后一个epoch，才暂停
#     #     print("Pausing for 30 minutes...")
#     #     time.sleep(1800)  # 暂停1800秒（即半小时
# # 绘制训练和验证的损失曲线
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(metrics["train_losses"], label="train loss")
# ax.plot(metrics["valid_losses"], label="valid loss")
# ax.set_xlabel("epoch")
# ax.set_ylabel("loss")
# ax.set_xticks(range(n_epochs))
# ax.legend()
# ax.grid()
#
# # 绘制训练和验证的准确率曲线
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(metrics["train_accs"], label="train accuracy")
# ax.plot(metrics["valid_accs"], label="valid accuracy")
# ax.set_xlabel("epoch")
# ax.set_ylabel("accuracy")
# ax.set_xticks(range(n_epochs))
# ax.legend()
# ax.grid()
#
# # 加载最佳模型并在测试集上评估
# model.load_state_dict(torch.load("transformerThreeRecallTest.pt"))
# test_loss, test_acc, test_recalls = evaluate(test_data_loader, model, criterion, device, output_dim)
#
# print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")
# for class_id, recall in test_recalls.items():
#     print(f"Class {class_id} Test Recall: {recall:.3f}")
#
#
# # 定义情感预测函数
# def predict_sentiment(text, model, tokenizer, device):
#     ids = tokenizer(text)["input_ids"]
#     tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
#     prediction = model(tensor).squeeze(dim=0)
#     probability = torch.softmax(prediction, dim=-1)
#     predicted_class = prediction.argmax(dim=-1).item()
#     predicted_probability = probability[predicted_class].item()
#     return predicted_class, predicted_probability
#
# # 使用情感预测函数进行预测
# text = "This film is terrible!"
# predict_sentiment(text, model, tokenizer, device)
# text = "This film is great!"
# predict_sentiment(text, model, tokenizer, device)
# text = "This film is not terrible, it's great!"
# predict_sentiment(text, model, tokenizer, device)
# text = "This film is not great, it's terrible!"
# predict_sentiment(text, model, tokenizer, device)
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
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

# Set random seed for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Read Excel file and convert to datasets.Dataset format
df = pd.read_excel("Excel_File/Review25kPos_600each.xlsx")
# df = pd.read_excel("ReviewSmallSizeForTest30.xlsx")
df['label'] = df['label'].map({'Positive': 0, 'Neutral': 1, 'Negative': 2})



# Separate majority and minority classes
df_majority = df[df.label==0]  # 正面情感
df_neutral = df[df.label==1]   # 中性情感
df_negative = df[df.label==2]  # 负面情感

# Upsample minority classes
df_neutral_upsampled = resample(df_neutral,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=seed) # reproducible results

df_negative_upsampled = resample(df_negative,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=seed)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_neutral_upsampled, df_negative_upsampled])

# Now df_upsampled is the balanced dataset which we can use for further steps
train_df, test_df = train_test_split(df_upsampled, test_size=0.2, random_state=seed)
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)

# Convert DataFrame to Dataset
train_data = Dataset.from_pandas(train_df)
valid_data = Dataset.from_pandas(valid_df)
test_data = Dataset.from_pandas(test_df)

# Initialize transformer model's tokenizer
transformer_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)


# Define a function for tokenization and numericalization
def tokenize_and_numericalize_example(example):
    output = tokenizer(example["ids"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {"ids": output["input_ids"].squeeze(0), "attention_mask": output["attention_mask"].squeeze(0)}


# Apply the function to the dataset
train_data = train_data.map(tokenize_and_numericalize_example)
valid_data = valid_data.map(tokenize_and_numericalize_example)
test_data = test_data.map(tokenize_and_numericalize_example)

# Set the data format to torch and select the needed columns
train_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
valid_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["ids", "attention_mask", "label"])

pad_index = tokenizer.pad_token_id


# Define batch data processing function
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


# Define data loader
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn,
                                              shuffle=shuffle)
    return data_loader


# Set batch size and get data loaders
batch_size = 8
train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


# Define transformer model
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


# Load transformer from pre-trained model
transformer = transformers.AutoModel.from_pretrained(transformer_name)
output_dim = len(train_data["label"].unique())  # Number of unique labels
freeze = False
model = Transformer(transformer, output_dim, freeze)


# Count model's trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

# Set optimizer
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

# 修改损失函数以增大中性和负向情感的权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([1.0, 2.25, 1.75], dtype=torch.float32).to(device) # Increase the weight for Neutral and Negative
criterion = nn.CrossEntropyLoss(weight=weights)

# Check for GPU availability

model = model.to(device)
criterion = criterion.to(device)


# Update get_metrics function to calculate accuracy and recall
def get_metrics(prediction, label, num_classes):
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / label.shape[0]

    # Initialize containers for true positives and false negatives
    true_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    for class_id in range(num_classes):
        true_positives[class_id] = ((predicted_classes == class_id) & (label == class_id)).sum()
        false_negatives[class_id] = ((predicted_classes != class_id) & (label == class_id)).sum()

    # Calculate recall for each class to avoid division by zero, add a small epsilon
    recalls = (true_positives / (true_positives + false_negatives + 1e-6)).tolist()

    return accuracy.item(), recalls


# Define train function
def train(data_loader, model, criterion, optimizer, device, num_classes):
    model.train()
    epoch_losses = []
    epoch_accs = []
    epoch_recalls = [[] for _ in range(num_classes)]  # List of lists to hold recalls for each class
    for batch in tqdm.tqdm(data_loader, desc="Training"):
        ids = batch["ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids, attention_mask=attention_mask)
        loss = criterion(prediction, label)
        accuracy, recalls = get_metrics(prediction, label, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy)
        for i, recall in enumerate(recalls):
            epoch_recalls[i].append(recall)
    # Calculate the average recall for each class
    average_recalls = [np.mean(recall) for recall in epoch_recalls]
    return np.mean(epoch_losses), np.mean(epoch_accs), average_recalls



# Define evaluate function
def evaluate(data_loader, model, criterion, device, num_classes):
    model.eval()
    epoch_loss, epoch_accuracy = 0, 0
    all_predictions, all_labels = [], []
    recalls = {class_id: [] for class_id in range(num_classes)}
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch['ids'].to(device)  # 修改这里
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            predictions = model(ids, attention_mask)  # 注意这里也使用了 'ids'
            loss = criterion(predictions, labels)
            accuracy, recall_per_class = get_metrics(predictions, labels, num_classes)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            for class_id, recall in enumerate(recall_per_class):
                recalls[class_id].append(recall)
                # 收集所有预测和标签，以便稍后计算混淆矩阵
                preds = predictions.argmax(dim=-1).cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()  # Convert labels to NumPy array only if it's a tensor
                all_predictions.extend(preds)
                all_labels.extend(labels)

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))

    return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader), {class_id: np.mean(recalls[class_id]) for class_id in range(num_classes)},cm



# Training and evaluation loop
n_epochs = 3
best_valid_loss = float("inf")

metrics = {"train_losses": [], "train_accs": [], "train_recalls": [], "valid_losses": [], "valid_accs": [],
           "valid_recalls": []}
# 为每个类动态添加召回率的键。
for i in range(output_dim):
    metrics[f"train_recall_class_{i}"] = []
    metrics[f"valid_recall_class_{i}"] = []
for epoch in range(n_epochs):
    train_loss, train_acc, train_recalls = train(train_data_loader, model, criterion, optimizer, device, output_dim)
    valid_loss, valid_acc, valid_recalls,cm = evaluate(valid_data_loader, model, criterion, device, output_dim)

    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    for i, recall in enumerate(train_recalls):
        metrics[f"train_recall_class_{i}"].append(recall)

    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    for i, recall in enumerate(valid_recalls):
        metrics[f"valid_recall_class_{i}"].append(recall)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "transformerThreeRecall_eachMovie.pt")
    print(
        f"Epoch: {epoch}:\nTrain Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train Recalls: {[round(recall, 3) for recall in train_recalls]}, \nValid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}, Valid Recalls: {[round(recall, 3) for recall in valid_recalls.values()]}")

    # The remaining parts of your original code related to plotting and testing can be included here without modification.

    # 暂停半小时
    if epoch < n_epochs - 1:  # 如果不是最后一个epoch，才暂停
        print("Pausing for 30 minutes...")
        time.sleep(600)  # 暂停1800秒（即半小时
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
model.load_state_dict(torch.load("transformerThreeRecall_eachMovie.pt"))
test_loss, test_acc, test_recalls,test_cm = evaluate(test_data_loader, model, criterion, device, output_dim)

print(f"\ntest_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")
for class_id, recall in test_recalls.items():
    print(f"Class {class_id} Test Recall: {recall:.3f}")
print(f"Test Confusion Matrix:\n{test_cm}")


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
