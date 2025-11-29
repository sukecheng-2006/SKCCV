# SKCCV 代码示例

该仓库包含基于 Fashion-MNIST 数据集的多层感知机（MLP）示例，分别展示了从零实现和使用 PyTorch 模块化接口的两种方式，便于对照学习。

## 环境依赖
- Python 3.9+
- [PyTorch](https://pytorch.org/) 及其依赖
- [d2l](https://d2l.ai/) 库（`pip install d2l`）
- matplotlib（用于显示训练曲线）

## 文件说明
- `main.py`：使用 `torch.nn.Sequential` 构建单隐藏层 MLP 的主脚本，训练并绘制损失/精度曲线。
- `多层感知机简洁实现.py`：与 `main.py` 共享相同的简洁实现，作为独立的入口脚本。
- `多层感知机从0实现.py`：手动定义参数与前向传播的 MLP，展示从零实现的细节。

## 运行方式
在安装完依赖后，可使用以下命令训练模型并查看训练过程：

```bash
python main.py
python 多层感知机简洁实现.py
python 多层感知机从0实现.py
```

默认配置中，三者都会加载 Fashion-MNIST 数据集，训练 10 个 epoch，并调用 `matplotlib` 展示训练曲线或预测结果。
