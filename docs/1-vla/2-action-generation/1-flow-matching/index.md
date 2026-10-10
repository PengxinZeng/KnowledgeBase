# Flow Matching

**代表工作**：π0

**核心思想**：训练时把真实动作 `actions` 与高斯噪声 `noise` 线性插值得到 `x_t = t·noise + (1-t)·actions`，让网络回归目标速度场 `u_t = noise - actions`；推理时从纯噪声出发，用 Euler 法迭代 `num_steps`（默认 10）步去噪，得到动作 chunk。

**要点**

- 时间步 `t` 按 Beta(1.5, 1) 采样后缩放到 `[0.001, 0.999]`，偏向更噪的时间步
- 损失：`mean(square(v_t - u_t), axis=-1)`，逐动作步的 MSE
- 推理时 prefix 只前向一次并填充 KV cache，去噪循环里只重算 suffix

**论文源文件**：待补充

## 笔记

（待补充）
