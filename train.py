import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.lightweight_cnn import get_model
from data.dataset import get_data_loaders
from utils.visualization import TrainingVisualizer, to_device
import time
import torch.amp as amp

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, class_names, scheduler):
    best_acc = 0.0
    visualizer = TrainingVisualizer()
    
    # 根据设备类型选择是否使用混合精度训练
    if device.type == 'cuda':
        scaler = amp.GradScaler()
        use_amp = True
    else:
        use_amp = False
        print("警告: 在CPU上训练，不使用混合精度训练")
    
    # 记录训练时间
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        batch_time = 0.0
        data_time = 0.0
        
        # 记录每个epoch的开始时间
        epoch_start_time = time.time()
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            # 记录数据加载时间
            data_time = time.time() - epoch_start_time
            
            # 将数据移动到设备
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            
            # 根据设备类型选择是否使用混合精度
            if use_amp:
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 记录批次处理时间
            batch_time = time.time() - epoch_start_time - data_time
            
            # 每100个批次打印一次训练信息
            if i % 100 == 0:
                print(f'Batch {i}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {torch.sum(preds == labels.data).item() / inputs.size(0):.4f} | '
                      f'Data Time: {data_time:.2f}s | '
                      f'Batch Time: {batch_time:.2f}s')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s')
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        all_scores = []
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = to_device(inputs, device), to_device(labels, device)
                
                # 根据设备类型选择是否使用混合精度
                if use_amp:
                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                        scores = torch.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    scores = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        val_time = time.time() - val_start_time
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Time: {val_time:.2f}s')
        
        # 更新可视化器
        visualizer.update(epoch_loss, val_loss, epoch_acc, val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            
        print()
        
        # 更新学习率
        scheduler.step(val_acc)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f'训练完成！总用时: {total_time:.2f}s')
    
    # 训练结束后生成所有可视化图表
    print("生成可视化图表...")
    
    try:
        # 基础图表
        visualizer.plot_training_curves()
        visualizer.plot_confusion_matrix(all_labels, all_preds, class_names)
        visualizer.plot_sample_predictions(model, val_loader, class_names, device)
        visualizer.plot_class_distribution(train_loader, class_names)
        visualizer.plot_model_architecture(model)
        
        # 新增图表
        visualizer.plot_roc_curves(np.array(all_labels), np.array(all_scores), class_names)
        visualizer.plot_precision_recall_curves(np.array(all_labels), np.array(all_scores), class_names)
        visualizer.plot_feature_distribution(model, val_loader, class_names, device)
        visualizer.plot_error_analysis(model, val_loader, class_names, device)
        visualizer.plot_classification_metrics(all_labels, all_preds, class_names)
        
        print("可视化图表生成完成！")
    except Exception as e:
        print(f"警告: 生成可视化图表时出现错误: {str(e)}")
        print("部分图表可能未能生成，但训练过程已完成。")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')
        print(f'可用GPU数量: {torch.cuda.device_count()}')
        print(f'当前GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB')
        print(f'当前GPU显存缓存: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB')
    else:
        print("警告: 未检测到GPU，将使用CPU进行训练")
        print("注意: CPU训练速度会显著慢于GPU训练")
    
    # 数据加载器
    data_dir = 'data/animals-10'
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=64)  # 增加批次大小
    
    # 获取类别名称
    class_names = sorted(os.listdir(os.path.join(data_dir, 'train')))
    
    # 模型
    model = get_model(num_classes=len(class_names))
    model = model.to(device)
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f'使用 {torch.cuda.device_count()} 个GPU训练!')
        model = nn.DataParallel(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 使用AdamW优化器
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # 检查是否存在已训练的模型
    if os.path.exists('best_model.pth'):
        print("发现已训练的模型，正在加载...")
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['best_acc']
        print(f"模型加载完成！最佳准确率: {best_acc:.4f}")
        
        # 直接进行可视化
        print("开始生成可视化图表...")
        visualizer = TrainingVisualizer()
        
        # 在验证集上进行评估
        model.eval()
        running_corrects = 0
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = to_device(inputs, device), to_device(labels, device)
                
                outputs = model(inputs)
                scores = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        val_acc = running_corrects.double() / len(val_loader.dataset)
        print(f'当前验证集准确率: {val_acc:.4f}')
        
        # 生成所有可视化图表
        try:
            visualizer.plot_confusion_matrix(all_labels, all_preds, class_names)
            visualizer.plot_sample_predictions(model, val_loader, class_names, device)
            visualizer.plot_class_distribution(train_loader, class_names)
            visualizer.plot_model_architecture(model)
            visualizer.plot_roc_curves(np.array(all_labels), np.array(all_scores), class_names)
            visualizer.plot_precision_recall_curves(np.array(all_labels), np.array(all_scores), class_names)
            visualizer.plot_feature_distribution(model, val_loader, class_names, device)
            visualizer.plot_error_analysis(model, val_loader, class_names, device)
            visualizer.plot_classification_metrics(all_labels, all_preds, class_names)
            
            print("可视化图表生成完成！")
        except Exception as e:
            print(f"警告: 生成可视化图表时出现错误: {str(e)}")
            print("部分图表可能未能生成，但训练过程已完成。")
    else:
        print("未找到已训练的模型，开始训练新模型...")
        # 训练模型
        num_epochs = 100  # 增加训练轮数
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, class_names, scheduler)

if __name__ == '__main__':
    main() 