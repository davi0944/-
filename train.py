import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import RobertaTokenizer, BatchEncoding, get_scheduler
from sklearn.model_selection import train_test_split
from model import MultimodalEmotionModel
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os
import chardet

# 数据集类
class MultimodalDataset(Dataset):
    def __init__(self, text_paths, image_paths, labels, tokenizer, transform=None):
        self.text_paths = text_paths
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.text_paths)

    def __getitem__(self, idx):
        # 自动检测文件编码
        with open(self.text_paths[idx], 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'

        # 加载文本
        with open(self.text_paths[idx], 'r', encoding=encoding, errors='ignore') as f:
            text = f.read().strip()
        text_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_input, image, label

# 自定义 collate_fn
def collate_fn(batch):
    text_inputs, images, labels = zip(*batch)
    input_ids = [item['input_ids'].squeeze(0) for item in text_inputs]
    attention_mask = [item['attention_mask'].squeeze(0) for item in text_inputs]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask}), images, labels

# 训练函数
def train_model(train_loader, model, criterion, optimizer, scheduler, device, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    for step, (text_input, image_input, labels) in enumerate(train_loader):
        input_ids = text_input['input_ids'].to(device)
        attention_mask = text_input['attention_mask'].to(device)
        image_input = image_input.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask, image_input)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(train_loader), correct / total

# 验证函数
def validate_model(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for text_input, image_input, labels in val_loader:
            input_ids = text_input['input_ids'].to(device)
            attention_mask = text_input['attention_mask'].to(device)
            image_input = image_input.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask, image_input)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(val_loader), correct / total

# 主训练函数
def main():
    # 强制使用 CPU
    device = torch.device('cpu')  # 设置为 CPU
    batch_size = 4
    num_epochs = 10
    learning_rate = 3e-5
    accumulation_steps = 4

    # 加载 RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像分辨率
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据
    with open('train.txt', 'r') as f:
        lines = f.readlines()[1:]

    text_paths, image_paths, labels = [], [], []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 2:
            guid, tag = parts
            text_paths.append(f"data/{guid}.txt")
            image_paths.append(f"data/{guid}.jpg")
            labels.append(tag)

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = [label_map[label] for label in labels]

    train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(
        text_paths, image_paths, labels, test_size=0.2, random_state=42)

    train_dataset = MultimodalDataset(train_texts, train_images, train_labels, tokenizer, transform)
    val_dataset = MultimodalDataset(val_texts, val_images, val_labels, tokenizer, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 创建模型
    model = MultimodalEmotionModel().to(device)

    # 冻结 RoBERTa 的部分层
    for param in model.text_model.encoder.layer[:-6].parameters():  # 冻结 RoBERTa 的前 6 层
        param.requires_grad = False

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

    best_val_accuracy = 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(train_loader, model, criterion, optimizer, scheduler, device, accumulation_steps)
        val_loss, val_accuracy = validate_model(val_loader, model, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_multimodal_emotion_model.pth')

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

if __name__ == '__main__':
    main()