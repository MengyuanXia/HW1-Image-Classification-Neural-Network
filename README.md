# HW1-Image-Classification-Neural-Network

本项目是一个用于训练和测试深度神经网络模型的代码。该模型用于对Fashion-MNIST数据集中的服装图像进行分类。

## 用法

### 环境要求

- Python 3.x
- NumPy
- scikit-learn
- matplotlib

### 数据集下载与准备

1. 下载Fashion-MNIST数据集并解压缩。[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
2. 将解压后的数据集文件放置在与`load_data()`函数中指定的文件夹下。

### 模型训练与测试

1. 运行model.py文件以开始模型训练和测试。训练过程将自动加载Fashion-MNIST数据集，将数据集分为训练集和验证集，并在训练过程中记录损失和验证准确率。

2. 在完成训练后，模型权重将保存为`best_model_weights.pkl`文件。您可以从以下链接下载训练好的模型权重：

   链接: https://pan.baidu.com/s/1sPkj6EAAwpPkBvNU69NOjw?pwd=4321

    ```bash
    python model.py
    ```

### 结果

当Learning Rate:0.1 Hidden Size:256 Regularization Strength:0.001时模型性能最佳，
结果为Validation Accuracy:86.65%	Test Accuracy:85.42%
具体如下：
| Learning Rate | Hidden Size | Regularization Strength | Validation Accuracy | Test Accuracy |
|---------------|-------------|-------------------------|----------------------|---------------|
| 0.01          | 64          | 0.001                   | 76.4%                | 75.6%         |
| 0.01          | 64          | 0.01                    | 79.1%                | 78.49%        |
| 0.01          | 128         | 0.001                   | 78.93%               | 78.07%        |
| 0.01          | 128         | 0.01                    | 79.92%               | 79.34%        |
| 0.01          | 256         | 0.001                   | 78.77%               | 78.45%        |
| 0.01          | 256         | 0.01                    | 81.25%               | 80.75%        |
| 0.1           | 64          | 0.001                   | 85.53%               | 84.67%        |
| 0.1           | 64          | 0.01                    | 79.72%               | 79.12%        |
| 0.1           | 128         | 0.001                   | 86.12%               | 85.07%        |
| 0.1           | 128         | 0.01                    | 79.42%               | 78.69%        |
| 0.1           | 256         | 0.001                   | 86.65%               | 85.42%        |
| 0.1           | 256         | 0.01                    | 79.58%               | 78.8%         |

