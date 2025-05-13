import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.lightweight_cnn import get_model

class AnimalRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        
    def initUI(self):
        self.setWindowTitle('动物识别系统')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # 创建图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        layout.addWidget(self.image_label)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 创建选择图片按钮
        self.select_button = QPushButton('选择图片')
        self.select_button.clicked.connect(self.selectImage)
        button_layout.addWidget(self.select_button)
        
        # 创建识别按钮
        self.recognize_button = QPushButton('识别')
        self.recognize_button.clicked.connect(self.recognizeImage)
        button_layout.addWidget(self.recognize_button)
        
        layout.addLayout(button_layout)
        
        # 创建结果显示标签
        self.result_label = QLabel('请选择一张图片进行识别')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(self.result_label)
        
        main_widget.setLayout(layout)
        
        # 初始化图片路径
        self.image_path = None
        
    def loadModel(self):
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(num_classes=10)
        
        # 加载预训练权重
        if os.path.exists('best_model.pth'):
            checkpoint = torch.load('best_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            print("模型加载成功！")
        else:
            print("错误：找不到模型文件 'best_model.pth'")
            
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 定义类别名称
        self.class_names = ['狗', '马', '大象', '蝴蝶', '鸡', 
                           '猫', '牛', '羊', '蜘蛛', '松鼠']
        
    def selectImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            # 显示图片
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.result_label.setText('图片已加载，点击"识别"按钮进行识别')
            
    def recognizeImage(self):
        if not self.image_path:
            self.result_label.setText('请先选择一张图片！')
            return
            
        try:
            # 加载和预处理图片
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            # 显示结果
            result_text = f'识别结果：{self.class_names[predicted_class]}\n'
            result_text += f'置信度：{confidence:.2%}'
            self.result_label.setText(result_text)
            
        except Exception as e:
            self.result_label.setText(f'识别过程出错：{str(e)}')

def main():
    app = QApplication(sys.argv)
    ex = AnimalRecognitionApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 