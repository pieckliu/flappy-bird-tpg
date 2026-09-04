# Flappy Bird：标准 TPG 与权重 TPG 对比实验

## 实验设计

`compare_experiment.py` 对每个训练随机种子成对运行两种算法：

- 完全相同的 `FlappyEnv`、训练布局、验证布局和最大步数；
- 不同训练种子相互独立；
- 所有模型最后在同一组、从未参与训练或选模的测试布局上评估；
- 保存每一代数据和每一个测试 episode，不只保存日志；
- 学习曲线默认按照累计训练环境步数对齐，因为 PyTPG 的实际 root-agent
  数量可能和请求的 `population` 不完全相等；
- 图中的学习曲线是多次训练的中位数和四分位区间，测试箱线图中的每个点
  代表一次完整训练，而不是把同一模型的 episode 当成独立训练样本。

## 安装画图依赖

```powershell
cd C:\code\flappy-bird-tpg
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
```

PyTPG 仍使用自己的隔离环境：

```powershell
cd C:\code\flappy-bird-tpg\pytpg_flappy
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 快速冒烟测试

先确认两个算法、CSV 汇总和 PNG 绘图都能完成：

```powershell
cd C:\code\flappy-bird-tpg
.\.venv\Scripts\python.exe compare_experiment.py `
  --runs 1 `
  --generations 2 `
  --population 10 `
  --episodes 1 `
  --validation-episodes 1 `
  --test-episodes 5 `
  --max-steps 500 `
  --output experiments\smoke
```

## 正式实验

建议至少 10 个独立训练种子；20 个会更可靠：

```powershell
.\.venv\Scripts\python.exe compare_experiment.py `
  --runs 20 `
  --generations 300 `
  --population 80 `
  --episodes 3 `
  --validation-episodes 5 `
  --test-episodes 100 `
  --max-steps 6000 `
  --output experiments\tpg_comparison
```

已有完整 seed 会自动跳过。某个 seed 中途失败后，使用 `--overwrite` 只删除
并重新执行本次命令涉及的 seed 目录。只重新汇总和画图时使用：

```powershell
.\.venv\Scripts\python.exe compare_experiment.py `
  --runs 20 `
  --generations 300 `
  --output experiments\tpg_comparison `
  --plot-only
```

`--plot-only` 的 `--runs`、`--base-seed` 和 `--seed-stride` 必须与原实验一致。

## 输出

输出目录包含：

- `weighted_tpg/seed_*/history.csv`：权重 TPG 每代指标；
- `standard_tpg/seed_*/history.csv`：标准 TPG 每代指标；
- 每个 seed 的 checkpoint、`test_episodes.csv` 和完整 `run.log`；
- `all_generations.csv`：全部学习曲线原始数据；
- `all_test_episodes.csv`：全部未见测试 episode；
- `summary.csv`：以一次训练为统计单位的均值、中位数、标准差和近似 95% CI；
- `comparison.png`：一张包含验证学习曲线和最终测试分布的对比图；
- `config.json`：可复现实验的参数和解释器路径。
