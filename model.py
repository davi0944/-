import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models


class MultimodalEmotionModel(nn.Module):
    def __init__(self, text_model_name='roberta-base', image_model_name='resnet50', num_classes=3):
        super(MultimodalEmotionModel, self).__init__()

        # 文本模型（RoBERTa）
        self.text_model = RobertaModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 512)  # 将文本特征映射到 512 维

        # 图像模型（ResNet-50）
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 512)  # 将图像特征映射到 512 维

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),  # 输入维度为文本和图像特征的拼接维度
            nn.Tanh(),
            nn.Linear(256, 1)  # 输出注意力权重
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # 输入维度为文本和图像特征的拼接维度
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, image_input, text_only=False, image_only=False):
        if text_only:
            # 仅使用文本特征
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 对应的特征
            text_features = self.text_fc(text_features)  # 映射到 512 维
            outputs = self.classifier(text_features)  # 分类器输入维度为 512
        elif image_only:
            # 仅使用图像特征
            image_features = self.image_model(image_input)  # 图像特征维度为 512
            outputs = self.classifier(image_features)  # 分类器输入维度为 512
        else:
            # 使用多模态特征
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 对应的特征
            text_features = self.text_fc(text_features)  # 映射到 512 维

            image_features = self.image_model(image_input)  # 图像特征维度为 512

            # 拼接文本和图像特征
            combined_features = torch.cat((text_features, image_features), dim=1)  # 拼接后维度为 1024

            # 注意力机制
            attention_weights = self.attention(combined_features)  # 计算注意力权重
            attention_weights = torch.softmax(attention_weights, dim=1)  # 归一化
            attended_features = combined_features * attention_weights  # 加权特征

            # 分类器
            outputs = self.classifier(attended_features)  # 分类器输入维度为 1024

        return outputs