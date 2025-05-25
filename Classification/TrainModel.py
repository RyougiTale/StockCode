import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# 本地模块导入
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../LSTM_Regression_Stock_Value')))
from Util.StockDataLoader import StockDataLoader
from Util.data_utils import prepare_data_for_pytorch_classification
from LSTM_Regression_Stock_Value.lstm_model import LSTMClassifier
from LSTM_Regression_Stock_Value.config import DATA_CONFIG, PREDICTION_CONFIG


def run_classification_training(stock_code=None, save_model_dir='./models/pytorch_lstm_classification'):
    print("开始执行PyTorch LSTM股市涨跌分类模型训练流程...")
    os.makedirs(save_model_dir, exist_ok=True)
    if stock_code is None:
        raise ValueError("stock_code is required")
    train_start_date = DATA_CONFIG['train_start_date']
    train_end_date = DATA_CONFIG['train_end_date']
    val_start_date = DATA_CONFIG['validation_start_date']
    val_end_date = DATA_CONFIG['validation_end_date']
    print(f"股票代码: {stock_code}")
    print(f"训练数据周期: {train_start_date} 到 {train_end_date}")
    print(f"验证数据周期: {val_start_date} 到 {val_end_date}")
    # 1. 加载股票数据
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    print(f"加载训练数据...")
    train_stock_data_df = loader.load_stock_data(stock_code, train_start_date, train_end_date)
    if train_stock_data_df is None or train_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的训练数据。训练中止。")
        return
    print(f"成功加载 {len(train_stock_data_df)} 条训练数据。")
    print(f"加载验证数据...")
    val_stock_data_df = loader.load_stock_data(stock_code, val_start_date, val_end_date)
    if val_stock_data_df is None or val_stock_data_df.empty:
        print(f"未能加载股票 {stock_code} 的验证数据。训练中止。")
        return
    print(f"成功加载 {len(val_stock_data_df)} 条验证数据。")
    # 2. 准备数据
    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG['n_past_days_debug']
    print(f"训练目标 - 用过去 {n_past_days} 天数据预测未来 {prediction_window} 天涨跌。")
    # 处理训练数据并拟合Scaler
    print("准备训练数据并拟合归一化...")
    X_train_orig, y_train_orig, feature_scaler = prepare_data_for_pytorch_classification(
        train_stock_data_df.copy(),
        n_past_days,
        prediction_window
    )
    if X_train_orig is None or y_train_orig is None:
        print("训练数据准备失败（可能数据不足）。训练中止。")
        return
    print(f"原始训练数据准备完成。X_train_orig shape: {X_train_orig.shape}, y_train_orig shape: {y_train_orig.shape}")
    print('原始训练集涨天数:', int(y_train_orig.sum()), '跌天数:', len(y_train_orig)-int(y_train_orig.sum()))
    
    # 保存scaler (基于原始未采样数据拟合)
    joblib.dump(feature_scaler, os.path.join(save_model_dir, f'{stock_code}_feature_scaler.pkl'))
    print(f"特征归一化器已基于原始训练数据拟合并保存到: {save_model_dir}")

    # 对训练数据执行欠采样 (Undersampling) 多数类
    X_train, y_train = X_train_orig.clone(), y_train_orig.clone() # 使用克隆以避免修改原始Tensor
    y_train_np = y_train.squeeze().cpu().numpy()
    
    unique_labels, counts = np.unique(y_train_np, return_counts=True)
    if len(unique_labels) == 2:
        label_counts = dict(zip(unique_labels, counts))
        minority_label = min(label_counts, key=label_counts.get)
        majority_label = max(label_counts, key=label_counts.get)

        minority_indices = np.where(y_train_np == minority_label)[0]
        majority_indices = np.where(y_train_np == majority_label)[0]
        
        n_minority = len(minority_indices)
        n_majority = len(majority_indices)

        if n_minority > 0 and n_majority > n_minority:
            print(f"对主训练集执行欠采样：少数类 ({minority_label}) 数量: {n_minority}, 多数类 ({majority_label}) 数量: {n_majority}")
            random_majority_indices = np.random.choice(majority_indices, size=n_minority, replace=False)
            
            balanced_indices = np.concatenate([minority_indices, random_majority_indices])
            np.random.shuffle(balanced_indices) 
            
            X_train = X_train[balanced_indices]
            y_train = y_train[balanced_indices]
            
            print(f"欠采样后主训练集大小: {len(X_train)}")
            print(f"欠采样后主训练集 '{int(minority_label)}' 类数量: {int((y_train == minority_label).sum())}, '{int(majority_label)}' 类数量: {int((y_train == majority_label).sum())}")
        elif n_minority == 0:
            print("警告: 主训练数据中没有少数类样本，无法进行欠采样。")
        elif n_majority <= n_minority:
             print("主训练数据多数类样本数量不大于少数类，无需欠采样。")
    else:
        print("警告: 主训练数据标签不是二分类或只有一类，跳过欠采样。")

    # 处理验证数据，使用已拟合的Scaler
    print("准备验证数据 (使用已拟合的归一化器)...")
    X_val, y_val, _ = prepare_data_for_pytorch_classification(
        val_stock_data_df.copy(),
        n_past_days,
        prediction_window,
        existing_feature_scaler=feature_scaler # 使用基于原始数据拟合的scaler
    )
    if X_val is None or y_val is None:
        print("验证数据准备失败（可能数据不足或与训练数据特征不一致）。训练中止。")
        return
    print(f"验证数据准备完成。X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print('验证集涨天数:', int(y_val.sum()), '跌天数:', len(y_val)-int(y_val.sum()))
    
    print(f"最终用于训练的集大小: {len(X_train)}")
    print(f"验证集大小 (来自独立时段): {len(X_val)}")
    
    # 4. 创建 DataLoader
    batch_size = PREDICTION_CONFIG['batch_size']
    train_dataset = TensorDataset(X_train, y_train) # 使用可能被欠采样过的 X_train, y_train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # 验证集不应 shuffle
    
    # 5. 定义并实例化模型
    if X_train.shape[2] != X_val.shape[2]: # 应该使用 X_train_orig 来获取原始特征维度
        print(f"错误：训练集特征维度 ({X_train_orig.shape[2]}) 与验证集特征维度 ({X_val.shape[2]}) 不匹配！")
        return
    input_dim = X_train_orig.shape[2] # input_dim 基于原始特征数量
    
    hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
    layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
    dropout = PREDICTION_CONFIG['lstm_dropout']
    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_size=1, dropout_prob=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型已实例化并移至设备: {device}")
    
    # 6. 定义损失函数和优化器
    learning_rate = PREDICTION_CONFIG['learning_rate']
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    best_val_model_path = os.path.join(save_model_dir, f'{stock_code}_val_best_lstm_classifier.pth')
    print(f"基于验证集最优的模型将尝试保存到: {best_val_model_path}")
    num_epochs = PREDICTION_CONFIG['num_epochs_debug']
    print(f"开始训练模型（包含验证），共 {num_epochs} 轮...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for batch_idx, (data, targets) in enumerate(batch_pbar):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_pbar.set_postfix({'Train Batch Loss': f'{loss.item():.4f}'})
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] 完成, 训练平均损失: {avg_epoch_loss:.6f}')
        
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        all_preds_epoch_val = []
        all_targets_epoch_val = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for val_data, val_targets in val_pbar:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                val_outputs = model(val_data)
                loss = criterion(val_outputs, val_targets)
                val_loss_epoch += loss.item()
                preds = (val_outputs > 0.5).float()
                correct += (preds == val_targets).sum().item()
                total += val_targets.size(0)
                all_preds_epoch_val.extend(preds.cpu().numpy())
                all_targets_epoch_val.extend(val_targets.cpu().numpy())
                val_pbar.set_postfix({'Val Batch Loss': f'{loss.item():.4f}'})
        avg_val_loss_epoch = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = correct / total if total > 0 else 0
        
        val_f1, val_precision, val_recall = 0, 0, 0
        if total > 0:
            targets_np_val = np.array(all_targets_epoch_val).ravel()
            preds_np_val = np.array(all_preds_epoch_val).ravel()
            val_f1 = f1_score(targets_np_val, preds_np_val, zero_division=0)
            val_precision = precision_score(targets_np_val, preds_np_val, zero_division=0)
            val_recall = recall_score(targets_np_val, preds_np_val, zero_division=0)

        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss_epoch:.6f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}')
        
        if val_acc > best_val_acc: # 或者可以基于 F1 分数保存模型
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_val_model_path)
            tqdm.write(f'*** Epoch {epoch+1}: 新的最佳验证模型已保存 (验证准确率: {best_val_acc:.4f}) ***')
    print("模型训练与验证完成。")
    if os.path.exists(best_val_model_path) and best_val_acc > 0:
        print(f"基于验证集最优的模型已保存在: {best_val_model_path} (最高验证准确率: {best_val_acc:.4f})")
    else:
        print("未能找到或保存更优的模型（基于验证准确率）。")
    print("PyTorch LSTM股市涨跌分类模型训练流程结束。")

def evaluate_single_features(stock_code=None, save_model_dir='./models/pytorch_lstm_classification'):
    print("\n===== 单特征有效性评估 =====")
    os.makedirs(save_model_dir, exist_ok=True)
    if stock_code is None:
        raise ValueError("stock_code is required")
    train_start_date = DATA_CONFIG['train_start_date']
    train_end_date = DATA_CONFIG['train_end_date']
    val_start_date = DATA_CONFIG['validation_start_date']
    val_end_date = DATA_CONFIG['validation_end_date']
    loader = StockDataLoader(data_dir=DATA_CONFIG['data_dir'])
    train_stock_data_df = loader.load_stock_data(stock_code, train_start_date, train_end_date)
    val_stock_data_df = loader.load_stock_data(stock_code, val_start_date, val_end_date)
    prediction_window = PREDICTION_CONFIG['prediction_days']
    n_past_days = PREDICTION_CONFIG['n_past_days_debug']
    # 扩展特征列表以包含默认的技术指标，因为 prepare_data_for_pytorch_classification 默认会计算它们
    # 如果只想严格测试单个原始特征，需要在 prepare_data_for_pytorch_classification 中调整不计算技术指标
    # 但当前 prepare_data_for_pytorch_classification 的 feature_columns 参数只控制使用哪些列，而不是计算哪些列
    base_feature_list = ['close', 'high', 'low', 'open', 'volume'] 
    # all_possible_features_for_single_eval = ['close', 'high', 'low', 'open', 'volume', 'ma5', 'ma10', 'rsi14']
    results = {}
    
    # 先获取一次包含所有默认特征的原始训练数据，用于拟合一个通用的scaler（如果需要单独评估技术指标）
    # 或者，为每个单特征独立拟合scaler，如下面的逻辑
    
    for feat_to_eval in base_feature_list: # 只评估基础特征
        print(f'\n==== 只用特征: {feat_to_eval} ====')
        
        # 准备数据，scaler会为当前单特征重新拟合
        X_train_orig_sf, y_train_orig_sf, feature_scaler_sf = prepare_data_for_pytorch_classification(
            train_stock_data_df.copy(),
            n_past_days,
            prediction_window,
            feature_columns=[feat_to_eval] # 传递单特征
        )

        if X_train_orig_sf is None or y_train_orig_sf is None:
            print(f"特征 {feat_to_eval} 数据准备失败，跳过。")
            results[feat_to_eval] = None
            continue
        
        # 对单特征的训练数据执行欠采样
        X_train_sf, y_train_sf = X_train_orig_sf.clone(), y_train_orig_sf.clone()
        y_train_np_sf = y_train_sf.squeeze().cpu().numpy()
        unique_labels_sf, counts_sf = np.unique(y_train_np_sf, return_counts=True)

        if len(unique_labels_sf) == 2:
            label_counts_sf = dict(zip(unique_labels_sf, counts_sf))
            minority_label_sf = min(label_counts_sf, key=label_counts_sf.get)
            majority_label_sf = max(label_counts_sf, key=label_counts_sf.get)

            minority_indices_sf = np.where(y_train_np_sf == minority_label_sf)[0]
            majority_indices_sf = np.where(y_train_np_sf == majority_label_sf)[0]
            
            n_minority_sf = len(minority_indices_sf)
            n_majority_sf = len(majority_indices_sf)

            if n_minority_sf > 0 and n_majority_sf > n_minority_sf:
                print(f"单特征 {feat_to_eval} 执行欠采样：少数类 ({minority_label_sf}) 数量: {n_minority_sf}, 多数类 ({majority_label_sf}) 数量: {n_majority_sf}")
                random_majority_indices_sf = np.random.choice(majority_indices_sf, size=n_minority_sf, replace=False)
                balanced_indices_sf = np.concatenate([minority_indices_sf, random_majority_indices_sf])
                np.random.shuffle(balanced_indices_sf)
                
                X_train_sf = X_train_sf[balanced_indices_sf]
                y_train_sf = y_train_sf[balanced_indices_sf]
                print(f"单特征 {feat_to_eval} 欠采样后训练集大小: {len(X_train_sf)}")
                print(f"单特征 {feat_to_eval} 欠采样后 '{int(minority_label_sf)}' 类: {int((y_train_sf == minority_label_sf).sum())}, '{int(majority_label_sf)}' 类: {int((y_train_sf == majority_label_sf).sum())}")
            elif n_minority_sf == 0:
                 print(f"警告: 单特征 {feat_to_eval} 训练数据中没有少数类样本，无法欠采样。")
            elif n_majority_sf <= n_minority_sf:
                 print(f"单特征 {feat_to_eval} 多数类样本不大于少数类，无需欠采样。")
        else:
            print(f"警告: 单特征 {feat_to_eval} 训练数据标签不是二分类或只有一类，跳过欠采样。")

        # 准备验证数据，使用该单特征训练数据拟合的scaler
        X_val_sf, y_val_sf, _ = prepare_data_for_pytorch_classification(
            val_stock_data_df.copy(),
            n_past_days,
            prediction_window,
            existing_feature_scaler=feature_scaler_sf, # 使用为当前单特征拟合的scaler
            feature_columns=[feat_to_eval]
        )

        if X_val_sf is None or y_val_sf is None: # 检查验证数据是否也成功生成
            print(f"特征 {feat_to_eval} 的验证数据准备失败，跳过。")
            results[feat_to_eval] = None
            continue
            
        batch_size = PREDICTION_CONFIG['batch_size']
        train_dataset_sf = TensorDataset(X_train_sf, y_train_sf)
        train_loader_sf = DataLoader(train_dataset_sf, batch_size=batch_size, shuffle=True)
        val_dataset_sf = TensorDataset(X_val_sf, y_val_sf)
        val_loader_sf = DataLoader(val_dataset_sf, batch_size=batch_size, shuffle=False) # 验证集不应 shuffle
        
        input_dim_sf = X_train_sf.shape[2] # 基于单特征的维度
        
        hidden_dim = PREDICTION_CONFIG['lstm_hidden_dim']
        layer_dim = PREDICTION_CONFIG['lstm_layer_dim']
        dropout = PREDICTION_CONFIG['lstm_dropout']
        model_sf = LSTMClassifier(input_dim_sf, hidden_dim, layer_dim, output_size=1, dropout_prob=dropout)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_sf.to(device)
        
        learning_rate = PREDICTION_CONFIG['learning_rate']
        criterion_sf = nn.BCELoss()
        optimizer_sf = optim.Adam(model_sf.parameters(), lr=learning_rate)
        
        num_epochs_sf = PREDICTION_CONFIG.get('num_epochs_single_feature_debug', 10) # 可配置的单特征训练轮数
        
        best_val_acc_sf = 0.0
        best_f1_sf = 0.0 
        best_precision_sf = 0.0
        best_recall_sf = 0.0

        for epoch in range(num_epochs_sf):
            model_sf.train()
            epoch_loss_sf = 0.0
            for data, targets in train_loader_sf:
                data, targets = data.to(device), targets.to(device)
                optimizer_sf.zero_grad()
                outputs = model_sf(data)
                loss = criterion_sf(outputs, targets)
                loss.backward()
                optimizer_sf.step()
                epoch_loss_sf += loss.item()
            avg_epoch_loss_sf = epoch_loss_sf / len(train_loader_sf) if len(train_loader_sf) > 0 else 0
            
            model_sf.eval()
            correct_sf = 0
            total_sf = 0
            all_preds_epoch_sf = []
            all_targets_epoch_sf = []
            val_loss_epoch_sf = 0.0
            with torch.no_grad():
                for val_data, val_targets in val_loader_sf:
                    val_data, val_targets = val_data.to(device), val_targets.to(device)
                    val_outputs = model_sf(val_data)
                    loss_val = criterion_sf(val_outputs, val_targets)
                    val_loss_epoch_sf += loss_val.item()
                    preds = (val_outputs > 0.5).float()
                    correct_sf += (preds == val_targets).sum().item()
                    total_sf += val_targets.size(0)
                    all_preds_epoch_sf.extend(preds.cpu().numpy())
                    all_targets_epoch_sf.extend(val_targets.cpu().numpy())
            
            avg_val_loss_epoch_sf = val_loss_epoch_sf / len(val_loader_sf) if len(val_loader_sf) > 0 else 0
            current_epoch_val_acc_sf = correct_sf / total_sf if total_sf > 0 else 0
            current_epoch_f1_sf, current_epoch_precision_sf, current_epoch_recall_sf = 0,0,0

            if total_sf > 0:
                targets_np_sf = np.array(all_targets_epoch_sf).ravel()
                preds_np_sf = np.array(all_preds_epoch_sf).ravel()
                current_epoch_f1_sf = f1_score(targets_np_sf, preds_np_sf, zero_division=0)
                current_epoch_precision_sf = precision_score(targets_np_sf, preds_np_sf, zero_division=0)
                current_epoch_recall_sf = recall_score(targets_np_sf, preds_np_sf, zero_division=0)
            
            if epoch == 0 or current_epoch_f1_sf > best_f1_sf : # 或者基于其他指标如F1
                 best_val_acc_sf = current_epoch_val_acc_sf
                 best_f1_sf = current_epoch_f1_sf
                 best_precision_sf = current_epoch_precision_sf
                 best_recall_sf = current_epoch_recall_sf

            if epoch == num_epochs_sf -1 : #打印最后一轮的指标
                 tqdm.write(f"特征 {feat_to_eval} - Epoch [{epoch+1}/{num_epochs_sf}] Train Loss: {avg_epoch_loss_sf:.4f} Val Loss: {avg_val_loss_epoch_sf:.4f} Acc: {current_epoch_val_acc_sf:.4f} F1: {current_epoch_f1_sf:.4f} P: {current_epoch_precision_sf:.4f} R: {current_epoch_recall_sf:.4f}")

        print(f"特征 {feat_to_eval} - 最佳验证 Acc: {best_val_acc_sf:.4f}, F1: {best_f1_sf:.4f}, Precision: {best_precision_sf:.4f}, Recall: {best_recall_sf:.4f}")
        results[feat_to_eval] = {'acc': best_val_acc_sf, 'f1': best_f1_sf, 'precision': best_precision_sf, 'recall': best_recall_sf}
        
        if y_val_sf is not None:
            num_val_samples = y_val_sf.shape[0]
            if num_val_samples > 0:
                num_positive_val_samples = int(y_val_sf.sum().item())
                print(f"特征 {feat_to_eval} - 验证集样本数: {num_val_samples}, 验证集正样本数: {num_positive_val_samples}")
            else:
                print(f"特征 {feat_to_eval} - 验证集数据为空。")
        else:
            print(f"特征 {feat_to_eval} - 验证集数据(y_val_sf)未生成。")
            
    print("\n===== 单特征评估结果汇总 =====")
    for feat, metrics in results.items():
        if metrics is not None:
            print(f"特征 {feat}: Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        else:
            print(f"特征 {feat}: 无数据")

if __name__ == "__main__":
    run_classification_training(DATA_CONFIG['default_stock_code'])
    evaluate_single_features(DATA_CONFIG['default_stock_code'])
