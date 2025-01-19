import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from model import MultimodalEmotionModel
from train import MultimodalDataset, collate_fn
import os

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8  # 根据显存可调整
model_path = 'best_multimodal_emotion_model.pth'  # 训练好的模型权重路径
val_file_path = 'val_split.txt'  # 验证集文件路径
data_dir = 'data'  # 数据文件所在的目录

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载验证集数据
def load_val_dataset(val_file_path, data_dir, tokenizer, transform):
    text_paths = []
    image_paths = []
    labels = []

    with open(val_file_path, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            line = line.strip()
            parts = line.split(',')
            if len(parts) == 2:  # 确保行中有 guid 和 tag 两部分
                guid, tag = parts
                guid = guid.strip()
                tag = tag.strip()

                # 检查标签是否有效
                if tag not in ['negative', 'neutral', 'positive']:
                    print(f"Warning: Invalid label '{tag}' in line: {line}")
                    continue  # 跳过无效标签

                # 生成文件路径
                text_path = os.path.join(data_dir, f"{guid}.txt")
                image_path = os.path.join(data_dir, f"{guid}.jpg")

                # 检查文件是否存在
                if not os.path.exists(text_path):
                    print(f"Warning: Text file not found {text_path}")
                    continue
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found {image_path}")
                    continue

                text_paths.append(text_path)
                image_paths.append(image_path)
                labels.append(tag)
            else:
                print(f"Skipping invalid line: {line}")  # 输出错误的行

    # 将标签转换为整数
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = [label_map[label] for label in labels]

    # 创建验证集数据集
    val_dataset = MultimodalDataset(text_paths, image_paths, labels, tokenizer, transform)
    return val_dataset

# 验证函数
def validate_model(val_loader, model, criterion, device, text_only=False, image_only=False):
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
            try:
                outputs = model(input_ids, attention_mask, image_input, text_only=text_only, image_only=image_only)
            except TypeError as e:
                print(f"Error in forward pass: {e}")
                raise

            loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return epoch_loss, accuracy

# 主函数
def main():
    # 加载验证集
    val_dataset = load_val_dataset(val_file_path, data_dir, tokenizer, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    model = MultimodalEmotionModel().to(device)

    # 加载权重文件
    checkpoint = torch.load(model_path, map_location=device)

    # 过滤掉不匹配的权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

    # 更新模型权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Model loaded successfully.")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 验证完整多模态模型
    val_loss, val_accuracy = validate_model(val_loader, model, criterion, device)
    print(f"Multimodal Model - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 验证仅文本模型
    val_loss_text, val_accuracy_text = validate_model(val_loader, model, criterion, device, text_only=True)
    print(f"Text Only Model - Val Loss: {val_loss_text:.4f}, Val Accuracy: {val_accuracy_text:.4f}")

    # 验证仅图像模型
    val_loss_image, val_accuracy_image = validate_model(val_loader, model, criterion, device, image_only=True)
    print(f"Image Only Model - Val Loss: {val_loss_image:.4f}, Val Accuracy: {val_accuracy_image:.4f}")

if __name__ == '__main__':
    main()
