import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torchvision.utils import make_grid
import os

def to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class TrainingVisualizer:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc):
        # 确保数据在CPU上
        if isinstance(train_acc, torch.Tensor):
            train_acc = train_acc.cpu().item()
        if isinstance(val_acc, torch.Tensor):
            val_acc = val_acc.cpu().item()
            
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
    
    def plot_training_curves(self):
        """绘制训练和验证的损失与准确率曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Training Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        # 增加验证集最高准确率的横线
        if self.val_accs:
            best_val_acc = max(self.val_accs)
            plt.axhline(best_val_acc, color='r', linestyle='--', label=f'Best Val Acc: {best_val_acc:.4f}')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_curves.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrix.png')
        plt.close()
    
    def plot_sample_predictions(self, model, dataloader, class_names, device, num_samples=16):
        """绘制样本预测结果"""
        model.eval()
        images, labels = next(iter(dataloader))
        images = images[:num_samples].to(device, non_blocking=True)
        labels = labels[:num_samples]
        
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # 创建图像网格
        grid = make_grid(images.cpu(), nrow=4, normalize=True)
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.permute(1, 2, 0))
        
        # 添加预测标签
        for i in range(num_samples):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i].cpu().permute(1, 2, 0))
            plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/sample_predictions.png')
        plt.close()
    
    def plot_class_distribution(self, dataloader, class_names):
        """绘制类别分布图"""
        class_counts = np.zeros(len(class_names))
        for _, labels in dataloader:
            for label in labels:
                class_counts[label] += 1
        
        plt.figure(figsize=(12, 6))
        plt.bar(class_names, class_counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/class_distribution.png')
        plt.close()
    
    def plot_model_architecture(self, model, input_size=(3, 64, 64)):
        """绘制模型架构图"""
        try:
            from torchviz import make_dot
            # 确保输入张量和模型在同一个设备上
            device = next(model.parameters()).device
            x = torch.randn(1, *input_size).to(device, non_blocking=True)
            y = model(x)
            dot = make_dot(y, params=dict(model.named_parameters()))
            try:
                dot.render(f'{self.save_dir}/model_architecture', format='png')
                print("模型架构图生成成功！")
            except Exception as e:
                print(f"警告: 无法生成模型架构图，请确保已安装 Graphviz 并添加到系统环境变量中。错误信息: {str(e)}")
                print("提示: 请访问 https://graphviz.org/download/ 下载并安装 Graphviz")
        except ImportError:
            print("警告: 未安装 torchviz 包，无法生成模型架构图")
            print("提示: 请运行 'pip install torchviz' 安装所需包")
    
    def plot_roc_curves(self, y_true, y_scores, class_names):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc
        plt.figure(figsize=(10, 8))
        
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/roc_curves.png')
        plt.close()
        
    def plot_precision_recall_curves(self, y_true, y_scores, class_names):
        """绘制精确率-召回率曲线"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        plt.figure(figsize=(10, 8))
        
        for i in range(len(class_names)):
            precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
            avg_precision = average_precision_score(y_true == i, y_scores[:, i])
            plt.plot(recall, precision, label=f'{class_names[i]} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/precision_recall_curves.png')
        plt.close()
        
    def plot_feature_distribution(self, model, dataloader, class_names, device):
        """绘制特征分布图"""
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                feats = model.features(inputs).view(inputs.size(0), -1).cpu()
                features.append(feats)
                labels.extend(targets.numpy())
        if not features or not labels:
            print("未提取到特征或标签，跳过t-SNE绘图")
            return
        features = torch.cat(features, dim=0).numpy()
        labels = np.array(labels)
        print("features shape:", features.shape)
        print("labels shape:", labels.shape)
        if features.shape[0] < 2:
            print("样本数太少，无法绘制t-SNE")
            return
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        plt.figure(figsize=(12, 8))
        for i in range(len(class_names)):
            idxs = labels == i
            plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=class_names[i], alpha=0.6)
        plt.title('Feature Distribution (t-SNE)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/feature_distribution.png')
        plt.close()
        
    def plot_error_analysis(self, model, dataloader, class_names, device):
        """绘制错误分析图"""
        model.eval()
        errors = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # 找出预测错误的样本
                mask = preds != targets
                if mask.any():
                    error_inputs = inputs[mask]
                    error_targets = targets[mask]
                    error_preds = preds[mask]
                    
                    for i in range(len(error_inputs)):
                        errors.append({
                            'image': error_inputs[i].cpu(),
                            'true': error_targets[i].item(),
                            'pred': error_preds[i].item()
                        })
        
        # 绘制错误样本
        if errors:
            plt.figure(figsize=(15, 15))
            for i, error in enumerate(errors[:16]):  # 最多显示16个错误样本
                plt.subplot(4, 4, i+1)
                plt.imshow(error['image'].permute(1, 2, 0))
                plt.title(f'True: {class_names[error["true"]]}\nPred: {class_names[error["pred"]]}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/error_analysis.png')
            plt.close()
        
    def plot_classification_metrics(self, y_true, y_pred, class_names):
        """绘制分类指标"""
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[class_name][metric] for class_name in class_names]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Classification Metrics')
        plt.xticks(x + width, class_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/classification_metrics.png')
        plt.close() 