# 多模态情感识别

本仓库实现了一个结合文本和图像数据进行情感分类的多模态情感识别模型。模型使用 RoBERTa 处理文本数据，ResNet-50 处理图像数据，并通过注意力机制融合两种模态的特征。
## 环境配置

本实现基于 Python3。运行代码需要以下依赖：

- torch==1.10.0

- torchvision==0.11.1

- transformers==4.12.0

- scikit-learn==0.24.2

- Pillow==8.4.0

- pandas==1.1.5

可以通过以下命令安装依赖：

```python
pip install torch torchvision transformers scikit-learn pandas Pillow
```

## 仓库结构
仓库包含以下重要文件:

```python
project/
├── data/
│   ├── ...... .jpg
│   ├── ...... .txt
├── bert_base_uncased/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── tf_model.h5
│   ├── flax_model.msgpack
│   ├── pytorch_model.bin
│   ├── model.onnx
│   ├── model.safetensors
│   ├── rust_model.ot
├── model.py            # 包含 MultimodalEmotionModel 的实现
├── train.py            # 用于训练模型的脚本
├── run.py              # 用于在测试数据上运行推理的脚本
├── ablation_study.py   # 用于对模型进行消融研究的脚本
├── train_split.txt
├── val_split.txt
├── train.txt
├── test_without_label.txt
├── best_multimodal_emotion_model.pth
├── multimodal_emotion_model.pth
└── requirements.txt  # 所需的库
```

## 模型架构
model.py 中的 MultimodalEmotionModel 类结合了 RoBERTa 和 ResNet-50 分别提取文本和图像特征，并通过注意力机制动态加权文本和图像特征，最后将加权后的特征传递给分类器。


## 执行流程
1.要训练模型，请运行以下命令：
```python
python train.py
```
该脚本将：
从 train.txt 加载训练数据。  
预处理文本和图像数据。  
使用指定的超参数训练模型。  
将最佳模型保存到 best_multimodal_emotion_model.pth。

2. 要在测试数据上运行推理，请使用以下命令：
```python
python run.py
```
该脚本将：
从 test_without_label.txt 加载测试数据。  
预处理文本和图像数据。  
从 best_multimodal_emotion_model.pth 加载训练好的模型。  
生成预测结果并保存到 predictions.csv。  

3. 要对模型进行消融研究，请运行：
```python
python ablation_study.py
```
该脚本将：
从 val_split.txt 加载验证数据。  
评估模型在三种模式下的性能：多模态、仅文本和仅图像。  
打印每种模式的验证损失和准确率。  

## 引用

本代码部分基于以下仓库：

- [Hugging Face Transformers](https://github.com/huggingface/transformers)

- [PyTorch](https://github.com/pytorch/pytorch)

- [Torchvision](https://github.com/pytorch/vision)


