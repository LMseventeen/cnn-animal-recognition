import os
import shutil
import random
from pathlib import Path

def prepare_dataset(source_dir, train_dir, val_dir, train_ratio=0.8):
    """
    将源数据集按照指定比例分割为训练集和验证集
    
    Args:
        source_dir: 源数据目录
        train_dir: 训练集目录
        val_dir: 验证集目录
        train_ratio: 训练集比例
    """
    # 确保目标目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取所有类别
    categories = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"找到 {len(categories)} 个类别: {categories}")
    
    # 处理每个类别
    for category in categories:
        print(f"\n处理类别: {category}")
        
        # 创建类别目录
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # 获取当前类别的所有图片
        category_dir = os.path.join(source_dir, category)
        images = [f for f in os.listdir(category_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 随机打乱图片列表
        random.shuffle(images)
        
        # 计算分割点
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"类别 {category} 共有 {len(images)} 张图片")
        print(f"训练集: {len(train_images)} 张")
        print(f"验证集: {len(val_images)} 张")
        
        # 复制训练集图片
        for img in train_images:
            src = os.path.join(category_dir, img)
            dst = os.path.join(train_dir, category, img)
            shutil.copy2(src, dst)
        
        # 复制验证集图片
        for img in val_images:
            src = os.path.join(category_dir, img)
            dst = os.path.join(val_dir, category, img)
            shutil.copy2(src, dst)

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(current_dir, 'raw-img')
    train_dir = os.path.join(current_dir, 'data', 'animals-10', 'train')
    val_dir = os.path.join(current_dir, 'data', 'animals-10', 'val')
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录 {source_dir} 不存在!")
        return
    
    print("开始准备数据集...")
    print(f"源目录: {source_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    
    # 准备数据集
    prepare_dataset(source_dir, train_dir, val_dir)
    
    print("\n数据集准备完成!")

if __name__ == '__main__':
    main() 