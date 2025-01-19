import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from transformers import BertTokenizer
from model import MultimodalEmotionModel
import chardet


# 自动检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(1000)  # 读取部分数据
        result = chardet.detect(raw_data)
        return result['encoding']


# 定义模型预测函数
def predict(text, image_path, model, device):
    # 加载 tokenizer 和图像预处理
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 文本处理
    text_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # 图像处理
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # 增加批量维度
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return "file_not_found"

    # 模型推理
    model.to(device)  # 将模型加载到设备
    model.eval()

    with torch.no_grad():
        text_input = {k: v.to(device) for k, v in text_input.items()}
        image = image.to(device)
        output = model(text_input['input_ids'], text_input['attention_mask'], image)
        _, predicted = torch.max(output, 1)

    # 返回预测标签
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[predicted.item()]


# 加载测试数据
test_df = pd.read_csv('test_without_label.txt', header=0)

# 确保第二列为 null，提取 guid
test_df['guid'] = test_df['guid'].astype(str)  # 将 guid 转为字符串（匹配文件名）
test_df['image_path'] = test_df['guid'].apply(lambda x: f"data/{x}.jpg")
test_df['text_path'] = test_df['guid'].apply(lambda x: f"data/{x}.txt")

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalEmotionModel()  # 初始化模型结构

# 加载权重文件
try:
    checkpoint = torch.load('multimodal_emotion_model.pth', map_location=device)

    # 如果权重文件的键名与模型不匹配，可以手动映射
    new_checkpoint = {}
    for key, value in checkpoint.items():
        if key.startswith('bert.'):  # 如果键名以 'bert.' 开头
            new_key = key.replace('bert.', 'text_model.')  # 替换为 'text_model.'
            new_checkpoint[new_key] = value
        elif key.startswith('resnet.'):  # 如果键名以 'resnet.' 开头
            new_key = key.replace('resnet.', 'image_model.')  # 替换为 'image_model.'
            new_checkpoint[new_key] = value
        else:
            new_checkpoint[key] = value

    # 加载权重
    model.load_state_dict(new_checkpoint, strict=False)  # strict=False 允许部分加载
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)


# 预测函数应用到每一行
def predict_row(row):
    # 自动检测编码并读取文本
    try:
        encoding = detect_encoding(row['text_path'])
        with open(row['text_path'], 'r', encoding=encoding, errors='ignore') as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"Text file not found: {row['text_path']}")
        return "file_not_found"
    except Exception as e:
        print(f"Error reading file {row['text_path']}: {e}")
        return "error"

    # 调用预测函数
    return predict(text, row['image_path'], model, device)


# 应用预测到每一行
test_df['predicted'] = test_df.apply(predict_row, axis=1)

# 保存结果
test_df[['guid', 'predicted']].to_csv('predictions.csv', index=False)

print("Prediction complete. Results saved to predictions.csv.")