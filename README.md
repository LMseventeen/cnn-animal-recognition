# 轻量化动物识别系统

基于卷积神经网络的轻量化动物识别系统，针对复杂场景下的动物图像识别任务进行了优化。包含训练脚本和图形界面应用程序。

## 项目特点

- 轻量级CNN模型设计，适合在资源受限环境下运行
- 针对类内差异大、背景干扰强等问题进行了优化
- 使用Animals-10数据集，支持10类常见动物的识别
- 包含完整的数据预处理、训练和评估流程
- 生成各类性能图片
- 模型在验证集准确率达到84%
- 提供图形界面应用程序，方便模型使用

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- PyQt5 5.15.4+
- 其他依赖见 requirements.txt

## 安装

1. 克隆项目：
```bash
git clone https://github.com/LMseventeen/cnn-animal-recognition.git
cd [项目目录]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集准备
1.数据集在raw-img文件夹中
2.将数据集分成训练集（80%）和验证集（20%）
```bash
python prepare_dataset.py
```
3. 确保数据集结构如下：
```
data/animals-10/
    ├── train/
    │   ├── dog/
    │   ├── cat/
    │   └── ...
    └── val/
        ├── dog/
        ├── cat/
        └── ...
```

## 使用方法

### 训练模型

1. 训练模型：
```bash
python train.py
```

2. 模型将保存在 `best_model.pth` 文件中

### 绘制性能图片
```bash
python train.py
```
![联想截图_20250512214030](https://github.com/user-attachments/assets/da1dda76-add6-41b3-8d6e-9fce4eb2ca2f)

### 使用图形界面应用程序

1. 运行图形界面程序：
```bash
python animal_recognition_app.py
```

2. 在图形界面中：
   - 点击"选择图片"按钮选择要识别的动物图片
   - 点击"识别"按钮进行识别
   - 查看识别结果和置信度
![联想截图_20250513111720](https://github.com/user-attachments/assets/96065c16-def0-426f-b0d7-53f38ba66ddf)


3. 支持的图片格式：
   - PNG
   - JPG
   - JPEG

4. 支持的动物类别：
   - 狗 (dog)
   - 马 (horse)
   - 大象 (elephant)
   - 蝴蝶 (butterfly)
   - 鸡 (chicken)
   - 猫 (cat)
   - 牛 (cow)
   - 羊 (sheep)
   - 蜘蛛 (spider)
   - 松鼠 (squirrel)

## 模型架构

- 3个卷积块，每个块包含：
  - 卷积层
  - 批归一化
  - ReLU激活
  - 最大池化
- 2个全连接层
- Dropout正则化

## 数据增强

- 随机水平翻转
- 随机旋转
- 颜色抖动
- 标准化

## 训练参数

- 批次大小：64
- 学习率：0.001
- 优化器：Adam
- 训练轮数：100
- 损失函数：CrossEntropyLoss

## 图形界面功能

- 图片预览
- 实时识别
- 显示识别结果和置信度
- 支持GPU加速（如果可用）
- 友好的错误提示 

