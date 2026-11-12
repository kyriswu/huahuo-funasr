# FunASR ECI GPU 离线语音服务设计

用户已确认：GPU 优先；模型内置镜像；HTTP 服务；支持离线转写、录音内说话人标注、声纹注册/核验/识别；附 ECI 部署示例。

## 架构

Linux amd64 + Python 3.11 + PyTorch CUDA 12.6 + 固定版本 FunASR。目标 ECI NVIDIA T4，可在兼容的 NVIDIA GPU 上运行。GPU 模式启动时严格检查设备，不自动回退 CPU。显式 `DEVICE=cpu` 仅用于诊断。

采用 FunASR 的 SeACo-Paraformer 中文模型、FSMN-VAD、CT-Transformer 标点及 CAM++ 192 维声纹。固定模型版本，构建时完整下载到 `/opt/models`，保存文件哈希；运行时仅接受本地路径，关闭更新检查和远程代码执行，不安装依赖或下载模型。断网启动和真实推理是交付验收项。

FastAPI 单进程监听 8000。每个实例同一时间只执行一项推理，繁忙时返回 429，防止模型可变配置和 GPU 显存竞争。上传音频通过 FFmpeg 统一为 16 kHz 单声道 float32。限制上传大小、解码时长、解码超时；不接受远端音频 URL。长录音由 VAD 分段和按音频时长组批识别。

## 接口与数据

- `POST /v1/transcriptions`：完整音频上传，返回文本、毫秒时间戳、录音内 `speaker_id`；可选声纹身份匹配。支持指定已知说话人数。
- `POST /v1/voiceprints/{speaker_id}`：单人音频注册或替换声纹，VAD 去静音后提取并归一化嵌入。
- `GET /v1/voiceprints`、`DELETE /v1/voiceprints/{speaker_id}`：声纹管理。
- `POST /v1/voiceprints/identify`：1:N 余弦相似度识别，低于阈值返回未知。
- `POST /v1/voiceprints/verify`：上传两个音频，1:1 声纹核验。
- `/healthz`、`/readyz`：进程和模型就绪探针。

身份匹配阈值可配置，返回余弦分数而非概率。阈值需结合真实业务音频校准。注册只接受干净单人录音；分离属于 diarization，不提取重叠人声音轨。

声纹以带模型标识的 JSON 原子写入 `/data/voiceprints`，便于 ECI 挂载 NAS，不存原始录音。模型标识包含快照摘要，避免不同模型混用。通过文件锁保护读写。一个库由一个实例写入；多副本应使用独立目录或外接专门的存储服务，不依赖 NAS 锁提供跨实例事务。

可选 Bearer token 由环境变量注入，健康探针无需鉴权。日志记录请求状态，不打印音频、文本、嵌入或凭证。

## 交付和验证

提供多阶段 Dockerfile、版本清单、模型下载与校验脚本、GPU Compose、本机/断网验收脚本、ACR 推送命令以及直接 ECI 创建请求与 ACK/虚拟节点 Pod 示例。推荐初始 T4 16 GB、4 vCPU、12 GiB 内存；这是起始配置，不承诺吞吐，需根据真实录音压测。

测试覆盖声纹相似度和未知拒绝、存储重载/模型隔离、上传错误及静音、健康/鉴权、实际单人和双说话人推理、真实声纹正反样本、容器重启持久化与 `--network none` 推理。若本机 GPU 与 T4 不同，记录实际验证型号和未验证的 ECI 边界。

## 取舍

选择内置模型，以镜像体积换取无需公网、启动不下载模型的部署方式。NAS 模型方案需要预配置外部文件；官方 runtime 衍生方案仍需补齐声纹注册和识别，因此当前用 Python SDK 统一实现三类能力。
