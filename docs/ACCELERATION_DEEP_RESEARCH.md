# Netcast TennisVision 端到端加速技术研究

## 执行结论

Netcast 当前的主要矛盾已经不是球模型速度。2026-09-15 的隔离缓存实测中，
`deemo3.mp4` 共 2,779 帧、200.3 秒，球模型循环达到 32.9 FPS；但报告仍需
484.09 秒，完整标注成片需 753.19 秒。Pass A 独占 414 秒，成片渲染独占
269 秒。结果说明真正的瓶颈已经转移到人物分割后处理、多次视频解码、串行阶段和
CPU 成片。[^1]

推荐路线不是继续堆叠“更快模型”，而是分三层处理：

1. **立即做结果等价的重构**：人物批次预取、低分辨率人物掩码延迟栅格化、取消报告
   阶段的逐帧全分辨率 PNG、复用身份识别帧、补齐阶段级计时。
2. **随后做共享视频流水线**：一次顺序解码，同时向球和人物分支供帧；固定机位数据、
   颜色转换和缩放只计算一次。上传文件仍可保留全局轨迹后处理，但不再重复读片。
3. **服务器阶段做 GPU 常驻媒体链路**：NVDEC 解码、CUDA 内存中的缩放和模型输入、
   NVENC 编码；并为多用户引入任务队列和动态批处理。

在当前 RTX 3050 Ti Laptop 4GB 上，前两层的合理目标是把同一机位 `deemo3` 的报告
从 8分04秒压到约 5～6 分钟，同时保持 `scene3d.json` 完全一致。要达到报告时间不高于
原片时长，预计还需要人物分支的结构性变化或更强的服务器 GPU。这个目标不能仅靠
TensorRT 或改编码 preset 达成。

## 研究边界与验收原则

研究以未来 NVIDIA GPU 服务器为目标，但所有方案先在当前 Windows 本机验证。输入视频
按原生帧率逐帧处理，30 FPS 是必须可靠支持的基线，不允许通过静默丢帧获得速度。球的
观测、击球、落点、球员归属是准确率合同；最终标注视频的像素压缩方式可以改变，但人眼
可见内容和时间戳不能改变。

“速度”必须拆成三个不同指标：

- **报告延迟**：从分析开始到结构化数据和交互报告可用；
- **成片延迟**：从分析开始到最终标注 MP4 封装完成；
- **稳态吞吐**：每秒可处理多少输入帧，决定延迟是否会随直播时长不断增长。

任何模型论文或仓库只报告神经网络 FPS，而没有包含解码、预处理、后处理、事件判断和
编码时，均不能与上述端到端指标直接比较。RacketVision 本身也是 BallTrack、
RacketPose、TrajPred 三阶段模块，并未给出 Netcast 功能范围下的端到端耗时。[^2]

## 当前管线的实测成本

| 阶段 | `deemo3` 实测 | 观察 |
|---|---:|---|
| 固定机位校验与球场恢复 | 4 秒 | 已不是瓶颈 |
| 球模型循环 | 32.9 FPS | 已超过该视频帧率，也超过30 FPS目标线 |
| Pass A 总计 | 414 秒 | 包含球、人物、全分辨率掩码和元数据 |
| 报告可用 | 484.09 秒 | 原片时长的2.42倍 |
| 成片渲染 | 269 秒 | 逐帧CPU绘制并经x264编码 |
| 全部完成 | 753.19 秒 | 原片时长的3.76倍 |

当前视频实际为 13.87 FPS，而不是30 FPS。相同时长的30 FPS视频会有约2.16倍帧数，
因此必须使用“每帧成本”和稳态 FPS 规划服务器，不能把这次2.42倍报告时间直接外推为
30 FPS生产表现。

本机补充微基准进一步定位了成本：OpenCV 完整顺序解码 2,779 帧约需27秒；典型双人
情况下，把人物模型的低分辨率掩码逐个放大到2880×1620、合并、膨胀并压成PNG，约需
39.8毫秒/帧，整片理论累计约111秒。单人约26.8毫秒/帧，四人训练场景约66.4毫秒/帧。
这些操作目前位于报告生成的关键路径上。[^1]

## 第一优先级：人物分支结果等价重构

### 1. GPU 推理与 CPU 掩码后处理重叠

当前人物循环先读取四帧、等待 YOLO 分割完成，再在主线程逐帧进行全分辨率掩码处理；
只有全部处理结束后才提交下一批 GPU 工作。CPU 和 GPU 因此交替空等。球模型使用有界
两批预取后，循环已从23.6 FPS提升到33.2 FPS，而且1,737帧候选逐项完全一致。人物分支
应采用相同结构：后台线程负责“解码下一批 + 模型推理”，主线程处理上一批结果。

CUDA 官方最佳实践要求用分块、异步传输和不同流重叠主机计算、数据传输与设备计算；
固定页内存可以提高并允许异步主机到设备传输。[^3] PyTorch 也提醒，只有在生产线程或
DataLoader中正确准备 pinned memory，`non_blocking=True` 才可能真正获益；在主线程临时
执行 `pin_memory()` 反而会抵消收益。[^4]

实施时先做普通有界队列，不立即引入自定义 CUDA stream。其优点是模型输入和执行顺序
完全不变，最容易证明结果等价。预期可隐藏一部分每帧约40毫秒的掩码后处理，优先级 P0。

### 2. 原生模型掩码存储，延迟到真正需要时再栅格化

Ultralytics 的分割结果已经提供原始二值 mask、像素多边形和归一化多边形，不要求立即
生成原视频尺寸的PNG。[^5] 当前下游绝大多数人物身份、场地位置和回合模式逻辑使用
`person_boxes`；完整轮廓主要有两个用途：判断少数接触候选与最终渲染时让球场线位于
人物背后。

建议在 Pass A 中保存每个人的原生 mask 或无损压缩的低分辨率 mask，并记录原始 shape、
box 和插值参数。只有以下两种场合再按现有顺序执行最近邻放大、仿射、合并和膨胀：

- 接触分类真正查询该帧时；
- 后台成片渲染该帧时。

这不是用矩形框替代轮廓，也不改变接触判断。它只是把大约111秒的典型双人全分辨率
工作从“报告前每一帧”移动到“少量事件帧 + 报告后的渲染”。必须用逐帧 mask 哈希证明
延迟栅格化与旧实现输出一致。预期可使报告提前70～110秒，优先级 P0。

### 3. 人物身份识别复用现有帧

当前 OSNet 身份阶段以每5帧采样一次的方式重新打开视频并读取人物 crop。Pass A 已经
持有同一批帧和人物框，可以在对应采样帧当场保存小尺寸 crop，或把它送入独立有界队列。
这可以取消一次完整读片，并减少约20～30秒解码及颜色复制成本。身份 embedding 和后续
匹配仍使用同一权重、同一 crop、同一顺序，因此可以做到数值等价。

### 4. 固定球场模式停止无效图像工作

固定机位启用时，每帧球场角点和 paint mask 已经固定，但 Pass A 仍对每张原图执行部分
灰度转换和元数据复制。应通过 profiler 证明哪些值已不再被读取，再删除无效工作，而不是
凭代码阅读猜测。PyTorch profiler 能同时记录 CPU 和 CUDA 活动，并导出时间线；Nsight
Systems 还可以追踪 CUDA 与 Video Codec SDK。[^6][^7]

## 第二优先级：一次解码、多个消费者

当前同一视频至少被球场采样、RacketVision背景建立、球顺序推理、人物推理、身份识别和
最终渲染分别打开。单次 OpenCV 全片读取约27秒，多次读取还伴随BGR复制、缩放和缓存抖动。

上传文件模式下，推荐的最小改造是：

```text
视频文件
  └─ 顺序解码器
      ├─ 原始帧有界队列 ── 人物分支
      ├─ 512×288帧环形缓冲 ── RacketVision四帧窗口
      ├─ 每5帧人物crop ── 身份分支
      └─ 时间戳/音频索引 ── 事件分支
                         └─ scene3d.json先发布
视频文件第二次读取 ── 最终渲染与编码
```

最终渲染必须等待全局轨迹，因此保留第二次读取是合理的；报告前的视觉分支应尽量共享第
一次读取。队列必须有上限并实施背压，禁止为了追求吞吐把整片原始帧存入内存。一张
2880×1620 BGR帧约13.3 MiB，30帧就接近400 MiB。

RacketVision 的180帧中值背景在推理开始前就需要建立。上传文件可以先做稀疏随机读取，
随后开始一次共享顺序解码；直播模式则必须单独设计短暂预热或固定机位背景档案，不能
未经回归直接把中值背景换成球场校准底图。

该层预计再节省30～60秒报告时间。它对单用户延迟有价值，但收益小于人物掩码重构；
优先级 P1。

## 第三优先级：硬件视频链路

NVIDIA 的 NVDEC 与 NVENC 是独立于 CUDA 计算核心的专用硬件，可在 CUDA 模型推理的
同时承担解码和编码。官方文档明确支持将解码帧直接用于CUDA推理，NVDEC还提供缩放、
裁剪和颜色转换。[^8][^9] 这比“FFmpeg先解到CPU BGR，再由PyTorch复制回GPU”更符合
服务器长期架构。

PyNvVideoCodec 是 NVIDIA 官方维护的 Python 接口，支持 Windows/Linux、顺序线程解码、
GPU device memory 和 CUDA Array Interface，可与 PyTorch 交换设备数据。[^10][^11]
建议先用它搭建独立验证器：只输出帧号、PTS和每帧哈希，对比 OpenCV 解码基线；确认
帧数、时间戳和色彩转换后，再接入模型。

本机已经安装的 FFmpeg 包含 `h264_nvenc`，但实测无法启动：当前驱动551.61只提供
NVENC API 12.2，而该 FFmpeg 构建要求API 13.1及610以上驱动。此问题不应通过静默退回
软件编码掩盖。短期选择是升级兼容驱动或使用匹配旧SDK的FFmpeg；PyNvVideoCodec官方
列出的Windows最低驱动为531.61，因此也值得直接验证。[^10]

NVENC的首轮目标不是声称“画质完全相同”，而是使用相同分辨率、帧率、时间戳和音频，
建立可接受的视觉质量门槛。硬件编码官方定位就是高吞吐，但preset和码率选择可能改变
压缩质量。[^12] 当前269秒成片阶段中还包含Python绘制，NVENC只能消除编码部分，不能
自动消除CPU绘图。

长期可把缩放、颜色转换、简单叠加和编码都保留在GPU内存。Video Codec SDK 13.1的
零拷贝转码方案允许NVDEC与NVENC共享CUDA array，减少GPU内部复制和context switch；
它适合作为服务器终局，而不是当前Windows原型的第一步。[^13]

## 模型运行时：何时使用 TensorRT、ONNX Runtime 与 CUDA Graph

### 人物模型

此前本机 TensorRT FP16 人物实验只把模型约40.5 FPS提升到41.5 FPS，却改变了下游
事件，因此已经正确回退。这个结果说明当前端到端瓶颈主要不在模型forward，也说明
不能把A100上的官方宣传倍数套到3050 Ti。

TensorRT仍可在完成掩码和解码重构后重新评估，但顺序必须是：

1. FP32 TensorRT，逐帧比较 box、mask 和置信度；
2. FP16 mixed precision，使用 Polygraphy/中间层检查数值；
3. 只有事件、落点和视觉回归全部通过后才能进入生产；
4. INT8目前不进入计划，因为它需要代表性校准集，准确率风险最高。

NVIDIA明确指出低精度层可能造成准确率问题，并建议与黄金输出逐层验证；提高特定层精度
通常会牺牲一部分性能。[^14] Ultralytics公布的A100 TensorRT数据证明FP16可能更快，
但硬件、模型版本和任务都与本机不同，只能证明“值得测”，不能作为收益承诺。[^15]

### 球模型

球模型循环已经达到32.9 FPS，暂时不是P0。其输入固定为 batch 4、15×288×512，适合
CUDA Graph 或 `torch.compile(mode="reduce-overhead")`。CUDA Graph通过把固定形状的
多次kernel launch录制为一个图来降低CPU启动开销，但要求静态shape和稳定内存地址。[^16]
应先在完整 `demo` 和 `deemo3` 上逐热图比较，而不是只比较最终最大连通域；微小热图
变化可能在0.5阈值附近改变候选出生。

ONNX Runtime只有配合 I/O Binding 才能公平测试。默认CUDA执行如果输入不在GPU或输出
未预分配，会在 `Run()` 内进行CPU/GPU复制，容易把复制时间误认为模型时间。[^17]

## 人物检测与跟踪：高收益但不是“无损重构”

Good-Tennis已经采用人物检测器加BoT-SORT/ByteTrack，并允许设置人物检测间隔；这说明
“检测 + 时序跟踪”是成熟开源方案的常见工程路线，但其仓库没有公布与Netcast落点合同
等价的端到端回归。[^18] ByteTrack通过保留低置信度框进行第二阶段关联来恢复遮挡目标，
论文在MOT17上报告29.6 FPS，但使用V100且指标是行人MOT，不代表网球击球识别速度。[^19]

Netcast曾直接把人物stride设为2，`demo`击球从35降到33，因此不能简单重复。下一次实验
应改为**自适应全率**：

- 每帧运行轻量人物box检测或运动跟踪；
- 分割mask只为绘制和少量接触候选生成；
- 球进入任何球员扩展邻域、轨迹反向或音频出现冲击时，强制前后窗口逐帧人物推理；
- 其他稳定时段才允许tracker维持人物状态；
- 对训练模式、双打和换边分别设置回归集。

DeepStream官方支持在非推理帧由tracker继续输出目标，并明确说明推理interval可以大于0；
但这只是能力，不是准确率保证。[^20] Ultralytics当前文档把ByteTrack定位为轻量快速基线，
BoT-SORT则增加ReID和相机运动补偿。固定机位、场上人数少的Netcast应先测ByteTrack或
OC-SORT，身份归属仍由已有OSNet负责。[^21]

该路线有望把人物计算降低40%～70%，是达到实时的关键，但属于算法变更，优先级P2，
不能与P0/P1的结果等价重构混在同一次提交。

## 流式与服务器架构

当前事件层依赖整片play mode、全局平滑、21帧落点窗口、回合恢复和身份注册，因此即使
所有模型超过30 FPS，也不会自动变成固定延迟直播。服务器版需要把输出分成两级：

- **暂定事件**：经过约0.7～2秒有限未来窗口即可显示；
- **确认事件**：回合结束或更长窗口后锁定，驱动黄色区域、落点和统计。

在线报告可以持续写入事件流；最终高质量成片仍在回合或视频结束后渲染。这样不会为了
直播强行删除现有全局审计逻辑。

单用户时，直接在一个进程内维护有界队列通常比立刻部署Triton简单。多用户服务器才应
引入Triton：其动态批处理可以合并不同请求以提高吞吐，instance group支持同卡或多卡
并发模型实例。[^22] 但动态批处理优化的是服务器吞吐，不保证单视频延迟下降；4GB本机
同时常驻多份模型还可能造成OOM。

建议的服务器数据流是：

```text
上传/RTSP → 解复用 → NVDEC → GPU帧池
                         ├─ 球时序批次 → 轨迹状态机
                         ├─ 人物检测/跟踪 → 身份与接触
                         └─ 低频球场验证
                                   ↓
                         延迟事件流与结构化报告
                                   ↓
                   GPU叠加/分段编码 → NVENC → HLS/MP4
```

## 分阶段实施计划与收益判断

| 阶段 | 改动 | 预计报告收益 | 准确率风险 | 进入条件 |
|---|---|---:|---|---|
| P0.1 | 人物GPU批次预取，与CPU mask处理重叠 | 30～80秒 | 极低 | box/mask逐帧一致 |
| P0.2 | 原生mask存储，按需无损重建 | 70～110秒 | 低 | 重建mask逐帧哈希一致 |
| P0.3 | ReID复用采样帧/crop | 20～30秒 | 极低 | embedding与身份一致 |
| P1.1 | 球、人物、ReID共享一次顺序解码 | 30～60秒 | 低 | 帧号、PTS、输入tensor一致 |
| P1.2 | PyNvVideoCodec/NVDEC验证 | 取决于分辨率 | 中 | 色彩、帧数、坐标回归通过 |
| P1.3 | NVENC后台成片 | 主要缩短成片 | 低至中 | 画质、音频、时间戳通过 |
| P2.1 | 人物检测+tracker+接触窗口全率 | 40%～70%人物成本 | 中高 | 所有标注窗口不漏击球 |
| P2.2 | TensorRT/CUDA Graph | 硬件相关 | 中 | 热图/mask和事件回归通过 |
| P3 | 增量事件与流式输出 | 改变产品延迟模型 | 高 | 暂定/确认事件协议完成 |

收益区间不是承诺，而是根据本机微基准和当前阶段占比给出的工程预算。P0全部完成后，
同机位`deemo3`报告达到约5～6分钟是合理目标；P1完成后有机会进一步接近原片时长的
1.3～1.7倍。要稳定达到不高于1倍，需要P2、服务器级GPU，或两者兼有。

## 回归与性能验收矩阵

每项优化必须同时通过性能门和准确率门：

1. **输入合同**：原帧数、FPS、PTS、画面方向和音轨一致；30/60 FPS都不得丢帧。
2. **模型合同**：球候选逐帧presence、xy、confidence；人物box、mask与身份embedding。
3. **轨迹合同**：轨迹帧数、分段、生命周期诊断和`scene3d.json`哈希。
4. **事件合同**：`demo`冻结的29落点/35击球、`deemo3`的8落点/4击球，以及本地人工标注
   时间窗逐项比较。
5. **视觉合同**：紫色轨迹、黄色区域、小地图、人物遮挡层抽样逐帧比较；编码改动使用
   SSIM/VMAF和人工复核，不要求压缩后字节相同。
6. **性能合同**：冷启动与热启动分开；报告、成片、每阶段wall time、GPU利用率、显存
   峰值、CPU占用、解码/模型/后处理FPS及P95批次延迟全部记录。
7. **稳定性合同**：中断恢复、重复上传、并发拒绝/排队、OOM自动回退和错误信息。

性能分析工具应先于更复杂的运行时替换。PyTorch profiler用于模型与拷贝，Nsight Systems
用于观察CPU线程、CUDA、NVDEC/NVENC是否真正重叠。没有时间线证据时，不应假设“两个
线程”或“两个CUDA stream”天然更快。

## 最终建议

下一次开发只做P0，不碰球轨迹和落点算法：

1. 为人物阶段增加细分计时与有界预取；
2. 将人物原生mask与全分辨率栅格化解耦；
3. 证明接触帧重建mask与旧版逐像素一致；
4. 复用ReID采样帧；
5. 完整无缓存重跑`demo`和`deemo3`。

这条路线最符合当前证据：它直接针对约111秒的可见CPU浪费和GPU/CPU串行等待，收益比
冒险换模型更确定，也最容易满足“准确率完全不损失”。完成后再决定是否进入NVDEC或
自适应人物tracker实验。

## Sources

[^1]: Netcast TennisVision，本地无缓存`deemo3`端到端基准与人物mask微基准，2026-09-15；记录见本文件及`CURRENT_ARCHITECTURE.md`，仅本地可访问。
[^2]: Dong et al., “[RacketVision: official repository](https://github.com/OrcustD/RacketVision),” AAAI 2026 release, accessed 2026-09-15.
[^3]: NVIDIA, “[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),” sections on pinned memory and asynchronous overlap, accessed 2026-09-15.
[^4]: PyTorch, “[A guide on good usage of non_blocking and pin_memory()](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),” accessed 2026-09-15.
[^5]: Ultralytics, “[Instance Segmentation with Ultralytics YOLO](https://docs.ultralytics.com/tasks/segment),” Results output, accessed 2026-09-15.
[^6]: PyTorch, “[torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html),” accessed 2026-09-15.
[^7]: NVIDIA, “[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/),” accessed 2026-09-15.
[^8]: NVIDIA, “[Video Codec SDK](https://developer.nvidia.com/video-codec-sdk),” accessed 2026-09-15.
[^9]: NVIDIA, “[NVDEC Video Decoder API Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html),” accessed 2026-09-15.
[^10]: NVIDIA, “[PyNvVideoCodec — Get Started](https://developer.nvidia.com/pynvvideocodec),” accessed 2026-09-15.
[^11]: NVIDIA, “[PyNvVideoCodec Decoder API](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html),” accessed 2026-09-15.
[^12]: NVIDIA, “[NVENC Video Encoder API Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/),” accessed 2026-09-15.
[^13]: NVIDIA, “[Video Codec SDK 13.1: Zero-Copy Transcode](https://developer.nvidia.com/blog/nvidia-video-codec-sdk-13-1-zero-copy-transcode-av1-b-frames-and-frame-accurate-seek/),” 2026.
[^14]: NVIDIA, “[Improving Model Accuracy — TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/accuracy-builder.html),” accessed 2026-09-15.
[^15]: Ultralytics, “[TensorRT Export Performance](https://docs.ultralytics.com/integrations/tensorrt),” accessed 2026-09-15.
[^16]: PyTorch, “[Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),” 2021.
[^17]: ONNX Runtime, “[I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html),” accessed 2026-09-15.
[^18]: yo-WASSUP, “[Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis/blob/main/README_en.md),” accessed 2026-09-15.
[^19]: Zhang et al., “[ByteTrack official repository and MOT benchmarks](https://github.com/FoundationVision/ByteTrack),” ECCV 2022.
[^20]: NVIDIA, “[DeepStream gst-nvtracker](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvtracker.html),” accessed 2026-09-15.
[^21]: Ultralytics, “[Multi-Object Tracking mode](https://docs.ultralytics.com/modes/track),” accessed 2026-09-15.
[^22]: NVIDIA, “[Triton Dynamic Batching and Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html),” accessed 2026-09-15.
