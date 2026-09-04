# PyTPG empty-Team 故障恢复

若日志以 `ValueError: max() arg is an empty sequence` 结束，表示上游 PyTPG
进入了一个没有可用 Learner 的 Team。使用 `run_comparison_safe.py` 继续实验；
它只对这一种异常返回动作 `0`，其他异常仍然终止 worker。

先把失败目录改名保留：

```powershell
Move-Item `
  -LiteralPath "C:\code\flappy-bird-tpg\experiments\tpg_comparison\standard_tpg\seed_0000200042" `
  -Destination "C:\code\flappy-bird-tpg\experiments\tpg_comparison\standard_tpg\seed_0000200042.failed"
```

然后使用原参数重新启动安全入口：

```powershell
cd C:\code\flappy-bird-tpg
.\.venv\Scripts\python.exe run_comparison_safe.py `
  --runs 10 `
  --generations 200 `
  --population 80 `
  --episodes 3 `
  --validation-episodes 5 `
  --test-episodes 100 `
  --max-steps 6000 `
  --output experiments\tpg_comparison
```

完整的已有 seed 会自动跳过；失败的标准 TPG seed 会从第 1 代重新训练。
不要对整个 10-seed 命令添加 `--overwrite`，否则所有已完成结果都会重跑。
