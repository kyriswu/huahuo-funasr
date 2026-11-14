# FunASR ECI GPU 离线语音服务

基于 [modelscope/FunASR](https://github.com/modelscope/FunASR) 构建面向阿里云 ECI / ECS Docker 的 Linux amd64 GPU 镜像。ASR、VAD、标点和 CAM++ 模型均内置，推理运行时可以完全断网。支持内存异步任务、带签名的完成回调和显式 ACK；回调需要可访问下游的网络。

A10 24 GB ECS 的 Docker 启动和异步接口完整示例见 [ECS 部署与回调教程](docs/ecs-a10-async.md)。CUDA Runtime 12.6、cuBLAS 和 cuDNN 已内置，不使用宿主机的 CUDA Toolkit；宿主机需 NVIDIA 驱动和 NVIDIA Container Toolkit。支持 CUDA 12.8 的驱动可以运行当前镜像。

支持中文文件转写、带时间戳的说话人标注，以及单人声纹注册、1:1 核验和 1:N 识别。`speaker_0` 之类的编号仅在一份录音内有效；启用声纹匹配后，另返回已注册的身份。说话人标注属于 diarization，不输出分离的重叠人声音轨。

## 构建与启动

构建需要公网下载 Python 依赖和约 2.2 GB 模型文件，构建机不需要 GPU。运行需要 NVIDIA GPU 和容器 GPU 支持。默认目标 T4 16 GB，建议容器至少 4 vCPU、12 GiB 内存，并根据长录音压测调整。CUDA 12.6 建议搭配 560 系列或更新的 NVIDIA 驱动。

```bash
cd /root/funasr-eci
docker build --platform linux/amd64 -t funasr-eci:gpu .

export API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker run -d --name funasr-gpu --gpus all \
  -p 127.0.0.1:8000:8000 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 \
  -e API_TOKEN -e DEVICE=cuda:0 \
  --mount type=volume,src=funasr-voiceprints,dst=/data \
  funasr-eci:gpu

curl --fail http://127.0.0.1:8000/readyz
```

也可 `docker compose up -d --build`，需要支持 `gpus` 配置的 Docker Compose 2.30+。启动时校验全部模型文件，加载成功后 `/readyz` 才返回 200。找不到 GPU 会直接启动失败；诊断时可以显式指定 `DEVICE=cpu`。

OpenAPI 文档：`http://127.0.0.1:8000/docs`，点击 **Authorize** 填入 token。默认单进程、单推理任务；忙时返回 429 和 `Retry-After: 2`。不要增加 Uvicorn workers，扩容时增加 GPU 实例。

## 异步转写、回调与 ACK

长录音建议调用 `POST /v1/tasks`，上传完成后返回 202 和 `task_id`。状态与结果接口分开：

| 接口 | 作用 |
| --- | --- |
| `POST /v1/tasks` | 提交转写，表单沿用转写参数，额外支持 `callback_url` |
| `GET /v1/tasks/{task_id}` | 只查询状态，不含解析结果，不消费 |
| `GET /v1/tasks/{task_id}/result` | 获取完整事件，可重复获取，不消费 |
| `POST /v1/tasks/{task_id}/ack` | 保存结果后发送 `{"event_id":"evt_..."}`，确认消费并清理 |

回调直接 POST 完整事件。下游保存结果并按固定 `event_id` 去重后返回 2xx，服务端自动清理，无需另发 ACK。主动获取结果时，业务保存成功后再 ACK。两条确认路径均幂等；已消费取结果返回 410，未完成返回 409，未知或重启前任务返回 404。

原始音频在解析结束后立即删除。未消费结果不设自动过期，包括回调重试耗尽的结果；排队、运行和待领取均计入容量，满额返回 429。任务和结果只存在当前进程及 tmpfs，重启后不恢复；声纹注册数据继续保存在 `/data/voiceprints`。已消费的简要状态默认保留一小时，最多 1024 条。

回调需配置 `CALLBACK_ALLOWED_ORIGINS`（例如 `https://business.example.com,http://10.0.0.8:8080`）和独立 `CALLBACK_SECRET`。来源匹配 scheme、host、port，禁止重定向。HMAC 验签、重试规则和 curl 示例见 [部署教程](docs/ecs-a10-async.md)。

## 调用

转写并自动估计说话人数：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/transcriptions \
  -H "Authorization: Bearer $API_TOKEN" \
  -F file=@meeting.mp3
```

已知人数可增加 `-F speaker_count=2`。返回 `text`、`timestamps_ms`、`segments` 和 `speakers`，所有音频位置均为**毫秒**；`duration_seconds` 与 `processing_seconds` 为秒。

注册声纹，建议使用 10–30 秒干净的单人音频。至少需要 3 秒被 VAD 检出的有效语音：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/voiceprints/zhangsan \
  -H "Authorization: Bearer $API_TOKEN" \
  -F file=@zhangsan.wav -F name=张三
```

再次注册相同 ID 默认返回 409，增加 `-F replace=true` 可替换。ID 使用 1–64 个英文字母、数字、下划线或连字符，以字母或数字开头；`identify`、`verify` 为保留名称。

1:N 声纹识别：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/voiceprints/identify \
  -H "Authorization: Bearer $API_TOKEN" \
  -F file=@unknown.wav -F threshold=0.5 -F top_k=5
```

低于阈值时，`matched=false` 且 `speaker_id=null`，不会强行分配最近的已知身份。`candidates` 保留候选分数，便于校准。

1:1 核验两个单人音频是否属于同一人：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/voiceprints/verify \
  -H "Authorization: Bearer $API_TOKEN" \
  -F reference=@zhangsan.wav -F query=@query.wav -F threshold=0.5
```

同时转写、分人并识别姓名：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/transcriptions \
  -H "Authorization: Bearer $API_TOKEN" \
  -F file=@meeting.wav -F identify_speakers=true -F threshold=0.5
```

在 `speakers[].voiceprint` 中读取身份结果，与 `segments[].speaker_id` 关联。声纹相似度为 **[-1, 1] 的原始余弦分数，不是概率**。默认 0.5 是起始参数，应使用业务中的同人/异人录音校准。短音频、噪声、重叠说话会影响聚类和身份匹配。

管理声纹：

```bash
curl --fail http://127.0.0.1:8000/v1/voiceprints \
  -H "Authorization: Bearer $API_TOKEN"
curl --fail -X DELETE http://127.0.0.1:8000/v1/voiceprints/zhangsan \
  -H "Authorization: Bearer $API_TOKEN"
```

## 模型与版本

| 组件 | 固定来源 / 版本 |
| --- | --- |
| FunASR | [be264e9](https://github.com/modelscope/FunASR/commit/be264e9382b4fef735536f3d278ee14514098a1a)，包版本 1.4.25 |
| PyTorch / torchaudio | 2.8.0 + CUDA 12.6 |
| ASR | [SeACo-Paraformer](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)，v2.0.9 |
| VAD | [FSMN-VAD](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)，v2.0.4 |
| 标点 | [CT-Transformer 中英标点](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large)，v2.0.4 |
| 声纹 | [CAM++](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)，v2.0.2，192 维 |

[model-lock.json](model-lock.json) 记录每个文件的大小和 SHA-256，[constraints.txt](constraints.txt) 固定 Python 依赖版本。构建时校验上游源码压缩包和模型；运行时再次校验本地模型，仅传入绝对路径，`disable_update=True`、`trust_remote_code=False`，无运行时下载或 pip 安装。完整 Python 依赖记录保存在镜像的 `/opt/python-environment.txt`，上游来源记录位于 `/opt/funasr-upstream.json`。

FunASR 代码遵循上游 MIT 许可；模型使用和再分发遵循各模型卡中的条款。模型卡和公开示例录音保留在镜像对应的模型目录中。

## 声纹持久化与 NAS

声纹存放在 `/data/voiceprints/*.json`，只保存嵌入和元数据，不保存上传录音。每条记录包含模型指纹；不兼容或损坏的库会报错，避免静默混用。

容器以 **UID/GID 10001:10001** 运行。挂载 NAS 前，在已挂载 NAS 的管理主机上为独立服务目录设置权限，例如：

```bash
sudo mkdir -p /mnt/nas/funasr
sudo chown 10001:10001 /mnt/nas/funasr
```

将 NAS 的这个目录挂载到 `/data`。同一声纹目录只分配给一个服务实例写入；多副本应使用独立目录或外接存储服务。容器内部的文件锁保护本实例并发请求，不提供跨 ECI 实例的事务协议。未挂持久卷时，ECI 重建会丢失注册数据。

## 发布到 ACR 并用于 ECI

当前异步/回调/ACK 版已发布到北京 ACR 的 `ai_service/funasr-eci:gpu-async-ack-20261114` 和 `:gpu`，两个标签对应同一已验收镜像，见 [新版 ACR 发布记录](docs/release-acr-20261114.md)。原 `gpu-20261112` 初版仍可按其固定 digest 使用，见 [初版发布记录](docs/release-acr-20261112.md)。重新发布时使用与部署环境同地域、可从 VPC 访问的 ACR 仓库：

```bash
export ACR_REGISTRY=huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com
export ACR_IMAGE="$ACR_REGISTRY/ai_service/funasr-eci:gpu-async-ack-20261114"
docker login "$ACR_REGISTRY"
docker tag funasr-eci:gpu "$ACR_IMAGE"
docker push "$ACR_IMAGE"
```

部署时优先使用推送后得到的 `仓库@sha256:...` digest。ECI 需要拉取镜像所需的 ACR 访问能力，**语音服务本身无需公网**；NAS 同样需要 VPC 内可达。

下方 ECI/ACK 配置保留初版固定 digest。当前 A10 ECS Docker 部署使用 [新版教程](docs/ecs-a10-async.md)，包含临时 tmpfs 和回调配置。

直接创建 ECI 的请求生成器（不会创建云资源）：

```bash
export ECI_REGION_ID=cn-beijing
export ECI_VSWITCH_ID=vsw-替换
export ECI_SECURITY_GROUP_ID=sg-替换
export ECI_IMAGE="$ACR_REGISTRY/ai_service/funasr-eci@sha256:1434d95d42806a9e748b7bd7dd2272740e8963d0d0823f0c089edea39bc68f60"
export ECI_NAS_SERVER=替换.nas.aliyuncs.com
export ECI_NAS_PATH=/funasr
# API_TOKEN 沿用前面生成的值。
# 如需显式镜像凭据，设置 ECI_REGISTRY_SERVER、ECI_REGISTRY_USER、ECI_REGISTRY_PASSWORD。
python3 scripts/render_eci.py --output artifacts/eci-request.json
```

生成文件权限为 0600，包含 API token 及可选镜像凭据，不应提交到版本库。默认规格 `ecs.gn6i-c4g1.xlarge`，T4 16 GB 显存；容器请求 4 vCPU、12 GiB 内存、1 GPU。请确认目标地域/可用区有该规格库存及配额，或用 `ECI_INSTANCE_TYPE` 指定当地可用的 GPU 规格。

模板显式选择 `tesla=570` 驱动，可通过 `ECI_GPU_DRIVER_VERSION` 调整。[ECI 官方 GPU 规格文档](https://help.aliyun.com/document_detail/2579486.html)在 2026 年 11 月列出 Tesla 570/580；SDK 中较旧的驱动说明不作为部署依据。宿主机驱动与镜像中的 CUDA/PyTorch 分开管理，无需在镜像里安装 NVIDIA 内核驱动。

审阅请求后，可通过阿里云 OpenAPI 或以下 SDK 脚本创建。**下面带 `--apply` 的命令会创建计费 GPU 实例。** 凭证由阿里云标准凭证链读取，不写入镜像：

```bash
python3 -m venv .venv-deploy
.venv-deploy/bin/pip install alibabacloud-eci20180808==2.1.0 alibabacloud-credentials
.venv-deploy/bin/python scripts/create_eci.py artifacts/eci-request.json --apply
```

在 ACK / ACK Serverless 中使用 ECI 虚拟节点时，参见 [deploy/ack-eci-pod.yaml](deploy/ack-eci-pod.yaml)。镜像已填写本次发布的固定 digest，配置 NAS PVC 和 Secret 后即可使用。示例仅提供 ClusterIP 服务，通过 VPC 内网或你的网关访问 8000 端口。

镜像内的 Docker HEALTHCHECK 主要用于本机；ECI/ACK 请使用示例中显式配置的 `/healthz` 和 `/readyz`。首次启动给模型校验/加载预留时间。长录音为同步请求，客户端和前置网关需配置足够的读取超时。

## 参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEVICE` | `cuda:0` | 默认必须有 GPU；`cpu` 为显式诊断模式 |
| `API_TOKEN` | 空 | 非空时 `/v1/*` 要求 Bearer token；探针不鉴权 |
| `DATA_DIR` | `/data` | 声纹目录的父目录 |
| `CPU_THREADS` | `4` | 推理和聚类 CPU 线程数 |
| `BATCH_SIZE_S` | `120` | ASR 动态组批总音频秒数，显存不足时减小 |
| `MAX_SPEAKERS` | `15` | 自动聚类人数上限，显式 `speaker_count` 优先 |
| `VOICEPRINT_THRESHOLD` | `0.5` | 默认原始余弦匹配阈值 |
| `MAX_UPLOAD_MB` | `256` | 每个音频文件大小上限 |
| `MAX_AUDIO_SECONDS` | `7200` | 转写音频最长秒数，超限拒绝 |
| `MAX_VOICEPRINT_SECONDS` | `120` | 每条注册/识别/核验音频最长秒数 |
| `MIN_VOICE_SECONDS` | `3` | 声纹所需最少有效语音秒数 |
| `DECODE_TIMEOUT_SECONDS` | `120` | FFmpeg 单次解码超时 |
| `MAX_PENDING_TASKS` | `16` | 上传中、排队、运行及未消费任务总上限 |
| `MAX_TASK_AUDIO_MB` | `1024` | 全部任务暂存原始音频总大小上限 |
| `MAX_TASK_RESULT_MB` | `16` | 单任务事件大小上限，超限生成小型失败事件 |
| `TASK_STATUS_TTL_SECONDS` | `3600` | 仅已消费简要状态的保留时间，未消费不适用 |
| `TASK_HISTORY_LIMIT` | `1024` | 已消费简要状态最大条数 |
| `CALLBACK_ALLOWED_ORIGINS` | 空 | 允许的回调来源，逗号分隔；配置后才允许回调 |
| `CALLBACK_SECRET` | 空 | 独立回调 HMAC-SHA256 密钥，不复用 API token |
| `CALLBACK_TIMEOUT_SECONDS` | `10` | 回调连接/读写超时 |
| `CALLBACK_MAX_ATTEMPTS` | `10` | 总发送次数上限，耗尽后结果继续保留 |
| `CALLBACK_RETRY_SECONDS` | `10` | 重试起始间隔，指数退避、最高一小时，带抖动 |

支持 FFmpeg 可解码的 WAV、MP3、FLAC、M4A、OGG 等格式；统一转换为 16 kHz 单声道。仅接受文件上传，不接受远端 URL。

## 验证

运行单元和接口检查（当前补丁版可用 `-f Dockerfile.release` 复用已验收基础层）：

```bash
docker build --target test -t funasr-eci:test .
docker run --rm --network none funasr-eci:test
```

断网运行真实 GPU 验收，使用镜像内的公开示例音频，覆盖双说话人转写、正反声纹核验、身份映射、静音处理和服务重启后的声纹持久化：

```bash
docker run --rm --gpus all --network none \
  --entrypoint python funasr-eci:gpu scripts/offline_smoke.py
```

验收脚本在 `/data` 下创建独立临时声纹库，结束后清理，不修改现有注册记录。实际构建、硬件、测试结果及 ECI 验证边界见 [docs/validation.md](docs/validation.md)。

异步版另有 47 项检查、真实 GPU 签名回调/ACK 和 Docker 重启验收，见 [异步版验收记录](docs/validation-async.md)。
