# 模型参数安装

模型不保存在 Git 仓库中。三个生产模型合计约 76.9 MB，首次安装时由脚本下载或重建，
后续分析只读取 `models/` 中的冻结文件，不训练、不改参数，也不会重复下载。

## 最简单的方式

新电脑首次安装时运行：

```powershell
.\setup.ps1
```

它会安装 Python 环境，然后自动准备模型。如果环境已经安装，只想重新下载模型：

```powershell
.\download_models.ps1
```

脚本会拒绝任何大小或 SHA-256 不匹配的文件。

## 三个模型从哪里来

| 本地文件 | 大小 | 安装方式 |
|---|---:|---|
| `models/racketvision_balltrack_state_v1.pt` | 45.4 MB | 下载固定提交中的 RacketVision `balltrack_best.pth`，校验源文件后，无损提取 `state_dict` |
| `models/bounce_classifier_production_v1.pkl` | 25.3 MB | 下载固定提交中的公开轨迹 CSV，以固定随机种子和固定 pickle 协议确定性重建 |
| `models/yolo11n-seg.pt` | 6.2 MB | 从 Ultralytics 官方 Release 直接下载 |

下载地址、上游提交、源文件哈希、最终文件哈希和许可状态全部记录在
`assets/manifest.json`，安装器实现在 `tools/install_assets.py`。不要从聊天记录、网盘
同名文件或旧实验目录手工复制未经校验的权重。

## 离线安装

在另一台已经安装成功的电脑上，把下面三个文件复制到一个普通文件夹：

```text
racketvision_balltrack_state_v1.pt
bounce_classifier_production_v1.pkl
yolo11n-seg.pt
```

然后在离线电脑执行：

```powershell
.\setup.ps1 -AssetSource D:\Netcast-TennisVision-models
```

或者环境已经建立时：

```powershell
.\download_models.ps1 -SourceDirectory D:\Netcast-TennisVision-models
```

离线文件同样必须通过清单中的哈希校验。

## 许可边界

自动下载解决的是技术复现，不等于自动解决商业许可。RacketVision 上游声明 MIT；
Ultralytics 权重及软件需要按 AGPL-3.0 或相应 Enterprise License 使用；落点参考数据的
再分发条件仍需在公开发布前复核。详见 `THIRD_PARTY_NOTICES.md` 和
`docs/PUBLICATION_CHECKLIST.md`。
