# LithoGRPO 论文复现（独立补充实现）

这是根据用户提供的 24 页论文补写的实现，不是作者尚未发布的训练源码或权重。保留原有 shotcount/adabox，在 `lithogrpo/` 增加完整的训练、采样和评估入口。当前交付是算法与工程流程复现；没有完成论文表 1 的数值复现。

## 环境

建议 Python 3.10–3.12。先按机器 CUDA 版本安装 PyTorch，再安装依赖：

```powershell
cd D:\graduate\project\LithoGRPO
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
```

以下示例用 `python` 代表所选环境中的 Python。Windows 可替换为 `.\.venv\Scripts\python.exe`。参考 shot counter 还需要 pandas；没有 OpenCV 时用 SciPy 做等价的 8 连通标记。Numba 缺失时 fast counter 使用相同算法的 Python 实现，速度会变慢。PuLP 固定 `<4`，使用自带 CBC 的版本；没有 PuLP 时用 SciPy/HiGHS 求解相同的二元集合覆盖 ILP。本次本机检查使用 HiGHS 后备求解器，CBC 路径未实际验证；推荐正式训练额外安装 numba 和 pulp。

## 数据与光学核

直接兼容 LithoBench 已栅格化的数据目录：

```text
data/MetalSet/
  target/00001.png       # 目标，灰度值 0/255
  pixelILT/00001.png     # 同名优化掩模，用于 pretrain/SFT
kernel/
  kernels/focus.pt       # [kh, kw, K]，复数频域 SOCS 核
  kernels/defocus.pt
  scales/focus.pt        # [K] 权重
  scales/defocus.pt
```

文件按名字排序，前 90% 用于 train、后 10% 用于 test；保存完整数据集和固定排序以避免划分变化。StdMetal/StdContact 等独立测试集用 `--split all`。PNG 按最近邻缩放至 512，不做随机数据增强（光学核可能不具旋转对称性）。代码不会下载或伪造真实数据和光学核，也不会把 GLP 文件自动当成 PNG。

光学前向参考 LithoBench `pylitho/exact.py`，用 PyTorch FFT 和原生 autograd 重建 SOCS：focus/nominal dose、focus/max dose、defocus/min dose。默认取 24 个核、阈值 0.225、sigmoid steepness 50、dose 1/1.02/0.98，来自 LithoBench `config/lithosimple.txt`，并非论文明确给出的全部参数。频域核不能当作空间 PSF 再 FFT。

参考源：

- https://github.com/laiyao1/LithoGRPO
- https://github.com/shelljane/lithobench/blob/main/pylitho/exact.py
- https://github.com/shelljane/lithobench/blob/main/lithobench/dataset.py
- https://github.com/shelljane/lithobench/blob/main/pyilt/evaluation.py
- https://github.com/shelljane/lithobench/blob/main/config/lithosimple.txt

## 三阶段训练

在仓库根目录执行，替换数据和 kernel 路径。配置保存论文 Appendix E/Table 5 的阶段参数，显式命令行参数覆盖 JSON。

```powershell
python -m lithogrpo train --config configs/metal_pretrain.json --data data/MetalSet --out checkpoints/metal --device cuda
python -m lithogrpo train --config configs/metal_sft.json --data data/MetalSet --out checkpoints/metal --checkpoint checkpoints/metal/pretrain_last.pt --kernel-dir kernel --device cuda
python -m lithogrpo baseline --data data/MetalSet --checkpoint checkpoints/metal/sft_last.pt --kernel-dir kernel --out checkpoints/metal/baseline.json --device cuda
python -m lithogrpo train --config configs/metal_rlft.json --data data/MetalSet --out checkpoints/metal --checkpoint checkpoints/metal/sft_last.pt --baseline checkpoints/metal/baseline.json --kernel-dir kernel --device cuda
```

ViaSet 替换为 `via_pretrain.json` / `via_sft.json` / `via_rlft.json`。StdContact 从 ViaSet 的 **pretrain** 权重开始 SFT，使用 `stdcontact_sft.json`，重新计算 baseline 后执行 RLFT。StdContact 的轮数在正文第 7 页写 50，Table 5 写 100；这里采用 Table 5 的 100，`--epochs 50` 可切换。

RLFT 默认每次 10 个布局，每个生成 6 个候选，共 60 个样本。`--microbatch 1` 用梯度累积实现批量更新，避免同时保留 60 个大网络计算图。采样轨迹暂存 CPU；策略更新先计算整条轨迹的概率比，再逐 transition 反向传播，控制显存占用。此实现是单进程单设备，不包含论文四卡并行训练与吞吐优化。

`--checkpoint` 默认只迁移模型参数，用于切换阶段；`--resume` 同时加载优化器、更新数和 CPU Torch RNG，从下一 epoch 开始。周期 checkpoint 若位于 epoch 中间，resume 会跳过该 epoch 的剩余批次；这不是逐样本精确恢复，尤其 CUDA/Python/NumPy RNG 未完整恢复。需要严格可重复实验时，从固定 seed 重跑阶段。

各阶段保存 `*_last.pt`、周期权重、参数 JSON 和逐更新 JSONL。RLFT 的目标优势在首次更新时通常使 loss 接近 0，这是组内中心化后的预期现象；应同时检查非零梯度、奖励变化和实际指标。

## 推理与评估

```powershell
python -m lithogrpo infer --checkpoint checkpoints/metal/rlft_last.pt --target data/MetalSet/target/00001.png --out results/example --device cuda
python -m lithogrpo evaluate --data data/MetalSet --split test --checkpoint checkpoints/metal/rlft_last.pt --kernel-dir kernel --shot-backend reference --out results/metal --device cuda
```

默认 ODE 一步推理；用 `--steps 2`、`5`、`10` 做表 2 对照。输出二值 mask、连续 mask（infer）、逐样本和均值 JSON（evaluate），以及 target/mask/resist 拼图。计时包含 ODE 随机噪声和模型采样，排除指标计算及文件写入；CUDA 显式同步，但没有 warm-up 的第一批会偏慢，不能直接作为论文吞吐结论。

训练奖励和 baseline 使用 `fast` shot；论文最终表格使用 `reference` shot，必须显式选择。两种计数允许的矩形重叠条件不同，不应把数字混在同一列。`--shot-size` 默认 512；比较时保持相同工作分辨率。

默认 EPE 为本地 Manhattan 边界法：根据 2048 nm 物理视场、15 nm 容差、40 nm 采样间隔检查内外法向探针；它是近似值，边角及离散化处理与 LithoBench 不完全相同。正式比较可在 baseline、RLFT、evaluate 中都加入：

```text
--epe-backend lithobench --lithobench-root D:/path/to/lithobench
```

该模式导入上游 `pyilt.evaluation`，把已模拟的 resist/target 最近邻上采样到 1 nm/像素后调用原 boundaries/epecheck；上游仓库及其 config/kernel/依赖须完整可用，部分版本导入时会初始化默认模拟器。该适配器尚未在本机真实 LithoBench 数据上验证。对于非 Manhattan、触碰图像边缘或空边界输入，上游代码本身也可能有边界限制。光学计算仍在 `--size` 分辨率进行；若基准要求全分辨率物理仿真，应设置一致分辨率重新训练/评估，而不能仅上采样指标。

## 论文对应关系和无法唯一还原的设置

| 论文 | 实现 | 说明 |
|---|---|---|
| Eq.1–5 | `physics.py` | SOCS、光刻胶 sigmoid、L2/PVB；SFT 用软图的像素平方和 |
| Eq.8–9 | `flow.matching_loss` | Gaussian noise 到 [0,1] mask 线性插值；flow 采用像素 MSE，再做批均值 |
| Eq.10–11 | `cli.train` | 从随机 t 外推终点，clamp [0,1] 后计算物理损失；flow 权重为 1−λL2−λPVB |
| Eq.12–14 | `advantages` / `normalized_reward` / `grpo_loss` | G=6，冻结 SFT 全训练子集平均指标为 baseline；无额外 KL 项 |
| Eq.15 | `flow.transition` / `sde_sample` | 按印刷公式使用加号漂移；a=0.1，Fourier 低通 colored noise |
| Eq.18–19 | `trajectory_log_ratio` | 全像素、全时间步 log probability **求和**，不以像素均值替代；独立 Gaussian 近似 |
| Appendix D | `model.py` | 4 层下采样、8 组 GN、SiLU、每层两个残差块、sinusoidal 时间嵌入 |
| Appendix E | `configs/*.json` | 阶段轮数、LR、batch size 和 SFT 权重 |

必须区分的重建选择：

1. 论文只称约 87M 参数，附录关于每个 UpLayer 上采样的文字描述与表中每级两个 UpLayer 的组合存在解释空间。这里每个 decoder stage 上采样一次，采用池化前的同分辨率 skip，默认 base=64、time_dim=256，共 **89,178,945** 参数；不是作者权重兼容结构。表中输入示例 256，而实验写 512；默认采用实验的 512，网络可接受 16 的倍数。
2. Eq.15 在 t=0 有奇点。默认先做 t=0→0.1 的确定性 warm start，再从 0.1 到 1 做 10 个 SDE 步。`--start-time`、`--sde-steps` 可调；论文没给这两个数值。warm start 固定，不计入策略梯度，优化的是条件随机轨迹概率。
3. 论文 SDE 声称与 ODE 同边缘分布，但 colored noise 与独立 Gaussian 的计算本身存在近似；本实现遵循论文公式，不额外宣称这种边缘分布等价已经证明。
4. colored noise 用 Gaussian Fourier 低通，默认 cutoff=0.1 cycles/pixel，以期望方差归一化；论文未公开滤波器、截止频率、归一化和时间编码尺度。
5. SFT 后 EPE 可能为零，Eq.13 无法直接除以零。默认 `--baseline-floor 1`；组 std 用总体标准差并加 1e-8。clip=0.2、AdamW(weight_decay=0)、grad clip=1、Adam 默认 betas 都属于补充选择。
6. 训练损失是软图的平方和，奖励/评估为二值 resist 的 L2、PVB 像素计数，与 LithoBench 评估惯例一致。采样保留连续原始轨迹，仅评估时阈值化；不会 clamp 随机状态后再错误计算 Gaussian 密度。
7. baseline 必须来自 SFT checkpoint，并在全训练子集上采一次 ODE 计算。论文没有明确逐布局还是全数据 baseline，这里选择全数据；baseline 保存来源权重和全部参数。RLFT 会检查关键的光刻和指标配置匹配。
8. 训练 split 不做数据增强、默认单设备、不支持作者未公开的 pretrained checkpoint 格式。论文表格的 seed 平均值需要独立跑 seed 1–4 并自行汇总；本机没有运行四卡完整训练。

## 已修复的原有 shot counter 问题

直方图栈里 `pos` 表示矩形左端点，原代码却用前一个栈元素的 `pos+1` 推导左边界，遇到高度 [2,1,2] 会生成包含背景的矩形。现改为标准单调栈继承左边界，并跳过零高度与重复高度。扫描线区间使用右开端点 e+1。补充空掩模返回 0、输入阈值化和有限性检查，以及 CBC 最优状态检查，避免把求解失败当成正常 shot count。reference counter 改为延迟导入，fast 模式不必加载 OpenCV/pandas；OpenCV 不可用时使用 SciPy 连通分量。adabox 的进程池增加 context manager 回收，避免每个连通分量泄漏进程，矩形分解算法保持不变。

## 小规模检查

```powershell
python -m unittest discover -s tests -v
python tools/smoke.py --out work/smoke
```

smoke 生成合成矩形、使用 base=8 的小网络、32×32 输入与合成 Gaussian 光学模型，串联 pretrain→SFT→baseline→RLFT→infer→evaluate。它只检查流程与梯度，不用于证明论文精度；脚本拒绝覆盖已有合成样本，重跑请使用新输出目录。

单元检查涵盖所有 512 个 3×3 掩模的候选矩形合法性/完整覆盖、若干 ILP 已知最优解、SOCS 频域计算与梯度、策略概率比和梯度、零方差优势及零 baseline、reference counter 的简单连通分量，以及消失/膨胀轮廓的 EPE。

### 本次执行记录（2026-10-16）

- 修改位置：`D:\graduate\project\LithoGRPO`；原仓库 HEAD：`f694498f1523efb89381fd9fb8984247f166095e`。没有提交 Git commit。
- 环境：Python 3.12.2、PyTorch 2.4.1+cpu、NumPy 2.0.2、SciPy 1.14.1，CUDA 不可用；本地 `.venv` 继承系统 Python 包。
- `python -m unittest discover -s tests -v`：7 项通过，最终运行约 1.97 秒；进程池清理修复后无此前的 ResourceWarning。
- 最新 `tools/smoke.py` 完整运行成功：8 个合成样本，32×32、base=8，pretrain/SFT/RLFT 各 2 次更新；包括冻结 baseline、一步推理和 test 评估。RLFT 的两次反向传播梯度均有限且非零。
- 同一小模型的 reference counter 评估也运行成功。fast/reference 的数值不同符合二者允许重叠规则不同的定义。
- `git diff --check` 通过。PuLP/Numba 下载失败，未绕过下载完整性校验；实际运行使用 SciPy/HiGHS 与非 JIT 矩形剪枝。
- 未执行：真实数据集训练、真实 SOCS 核验证、原生 LithoBench EPE 适配器运行、CBC 后端运行、四 GPU/四 seed 实验、论文表格数值与速度复现。

完整的正式复现下一步需要提供 MetalSet/ViaSet/StdMetal/StdContact 栅格数据、对应 `kernel/` 目录和可训练的 CUDA 环境。可以先在小规模真实数据上检查光学核的能量尺度、分辨率与原生评估，再启动上述阶段配置。
