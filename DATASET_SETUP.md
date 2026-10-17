# LithoBench 数据安装

官方来源：https://github.com/shelljane/lithobench

已安装的光学核位于 `kernel/`，包含 focus/defocus 及配套 scales，来自上游 commit `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6`。

官方 benchmark 原始 GLP 版图位于 `data/benchmark/`，保留 ICCAD2013、OpenROAD、StdMetal、StdContact 的原始目录。它们不是优化掩模标签，不能替代 `pixelILT/`。

已按上游 `getTargets` 的放置规则生成 2048×2048 的评估 PNG，使用 Pillow 绘制整数 Manhattan 多边形，分别位于 `data/evaluation/MetalSetTest/target/`（10 张）、`ViaSetTest/target/`（10 张）、`StdMetal/target/`（271 张）、`StdContactTest/target/`（165 张）。这些目录可直接用于当前代码的 `evaluate --split all`，不用于配对预训练。原始 benchmark 和生成 PNG 的记录位于 `data/benchmark_manifest.json`；尚未与 OpenCV 逐像素比对。

训练数据包为官方 Google Drive 的 `lithodata.tar.gz`，大小 **14,696,141,190 字节**。后台程序支持分块重试和断点续传，下载完成后自动提取 `target/`、`pixelILT/`、`glp/`，保留原子集名称，不下载作者预训练模型。

## 查看进度

```powershell
cd D:\graduate\project\LithoGRPO
Get-Content data/install_status.json
Get-Content work/lithobench-download/download.log -Tail 10
```

只有当 `stage` 为 `complete` 时，训练数据才安装完成。`downloading` 表示仍在下载；`extracting` 表示正在解压；`failed` 会记录原因。下载进度按已完成、可断点复用的块计算；接收但尚未保存为完整块的数据不计入百分比。

完成后 `data/download_manifest.json` 列出实际子集、目标图数量、优化掩模数量及配对数量；以这个清单为准。示例：

```text
data/MetalSet/target/*.png
data/MetalSet/pixelILT/*.png
data/ViaSet/target/*.png
data/ViaSet/pixelILT/*.png
```

如果包内同时包含 StdContact 和 StdContactTest，会原样保留，避免混合训练与测试样本。原始 GLP 评估目录与栅格训练数据分开放置。

## 中断后继续

确认原下载进程已退出，再运行：

```powershell
.\.venv\Scripts\python.exe -u tools/fetch_lithobench.py --workers 12
```

不要同时启动多个下载程序。临时压缩包和断点记录在 `work/lithobench-download/`。`.part` 是稀疏文件，文件逻辑大小不代表已下载的字节数。

下载完成并验证配对关系、PNG 样本和 gzip 完整性后，程序自动退出。网络错误会重试，最终失败时保留完整块以便重启续传。程序不会启动训练。

开始 MetalSet 预训练：

```powershell
.\.venv\Scripts\python.exe -m lithogrpo train --config configs/metal_pretrain.json --data data/MetalSet --out checkpoints/metal
```

当前本机 PyTorch 为 CPU 版本；正式训练建议转到 CUDA 环境。后台下载使用 `requests`，如果另一台机器缺少此包，需要先安装它。
