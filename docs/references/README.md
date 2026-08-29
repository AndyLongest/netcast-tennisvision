# Netcast TennisVision 本地参考文献

这里保存直接影响当前算法设计的引用、公开原文链接和实现映射。论文PDF仅可作为开发者的
本地缓存，已被Git忽略；不要在未确认再分发权时上传。修改实现时，应同时在“项目采用”栏
说明适用条件，并用完整样例回归验证。

## 轨迹与单目三维

### Where Is The Ball（CVPRW 2025）

- 原文：https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.html
- 代码页：https://where-is-the-ball.github.io/
- 项目采用：二维观测表示为相机射线；在球场坐标系中估计高度并做重投影。
- 没有照搬：论文的LSTM依赖合成训练集和预训练权重，当前仓库没有对应数据与权重。

### Tennis ball trajectory decomposition（Sports Engineering 2026）

- DOI：https://doi.org/10.1007/s12283-026-00547-6
- 项目采用：按“击球/触地之间的单段飞行”建模；三维位置、速度、重力是基础状态。
- 后续方向：有可靠长轨迹或多相机数据后，再估计阻力、升力与旋转。
- 当前没有强行采用：单目30fps的5–9个短时观测不足以稳定辨识旋转轴和Magnus参数。

### TT3D（CVPRW 2025）

- 原文：https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Gossard_TT3D_Table_Tennis_3D_Reconstruction_CVPRW_2025_paper.html
- 项目采用：相机标定、动力学约束和重投影误差共同验证轨迹，而不是只在像素平面拟合。
- 没有照搬：TT3D使用乒乓球参数和数据，不能直接套用到网球。

## 当前实现映射

| 论文概念 | 本地实现 | 保护条件 |
|---|---|---|
| 像素点转相机射线 | `tracking/geometry.py` | 球场单应性和相机姿态必须有效 |
| 重力约束三维飞行 | `tracking/ballistics.py` | 至少5个真实观测，5–9点鲁棒拟合 |
| 三维点重投影验证 | `tracking/ballistics.py` | 中位数和P90像素误差门限 |
| 接触分段 | `tracking/world_tracker.py` | 球员击球和bounce边界不跨段 |
| 不确定时回退 | `tracking/ballistics.py` | 只修复缺失帧，不参与真实检测接纳 |

## 完整性校验

如果开发者按各来源许可自行下载本地PDF，可在PowerShell运行：

```powershell
Get-ChildItem docs/references/*.pdf | Get-FileHash -Algorithm SHA256
```

已审阅版本的SHA-256记录在 [SHA256SUMS.txt](SHA256SUMS.txt)，但校验表不代表再分发授权。引用元数据在
[references.bib](references.bib)。
