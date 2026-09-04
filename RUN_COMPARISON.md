# 运行 TPG 对比实验

请使用 `run_comparison.py` 作为公开入口；它兼容 Matplotlib 3.x 中箱线图
参数名的变化。完整实验设计与输出字段见 `EXPERIMENT.md`，其中命令里的
`compare_experiment.py` 替换为 `run_comparison.py`。

```powershell
cd C:\code\flappy-bird-tpg
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt

# 快速检查
.\.venv\Scripts\python.exe run_comparison.py `
  --runs 1 --generations 2 --population 10 `
  --episodes 1 --validation-episodes 1 --test-episodes 5 `
  --max-steps 500 --output experiments\smoke

# 正式实验
.\.venv\Scripts\python.exe run_comparison.py `
  --runs 20 --generations 300 --population 80 `
  --episodes 3 --validation-episodes 5 --test-episodes 100 `
  --max-steps 6000 --output experiments\tpg_comparison
```

只重新生成汇总 CSV 和图片：

```powershell
.\.venv\Scripts\python.exe run_comparison.py `
  --runs 20 --generations 300 `
  --output experiments\tpg_comparison --plot-only
```
