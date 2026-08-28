# Flappy Bird with Tangled Program Graphs

一个从零实现的小型实验项目：用 Tangled Program Graph（TPG）通过进化学习控制 Flappy Bird。

## 快速开始

```powershell
cd C:\code\flappy-bird-tpg
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --generations 100 --population 80
python play.py --model checkpoints\best.json
```

训练不需要图形窗口；`play.py` 使用 Pygame 展示最佳个体。首次训练可先用较小参数验证：

```powershell
python train.py --generations 5 --population 20 --episodes 2
```

## 状态、动作与奖励

- 状态：小鸟纵坐标、竖直速度、下一根管道的水平距离、上边缘、下边缘。
- 动作：`0` 不操作，`1` 扇动翅膀。
- 奖励：每存活一步 `+0.1`，穿过管道 `+5`，碰撞 `-10`。

## 这里的 TPG

每个 Team 包含若干带权重的 Program。Program 根据状态产生 bid，bid 最大者决定动作，或跳转到另一个 Team。多个 Team 因此组成一个有向图；执行时设访问集合和最大深度来防止环路。进化过程保留精英，并对 Program 权重、动作/Team 引用和 Team 结构做变异。

这是面向学习的紧凑实现，不依赖第三方 TPG 框架，适合继续加入 crossover、共享子图、lexicase selection 或更复杂的寄存器程序。

## 测试

```powershell
python -m unittest discover -s tests -v
```

训练结果保存在 `checkpoints/`，该目录中的模型默认不提交到 Git。
