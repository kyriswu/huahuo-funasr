# 构建与验收记录

日期：2026-11-12。验收对象为本机 Docker 镜像，未创建阿里云计费资源。

## 镜像

- 标签：`funasr-eci:gpu`、`funasr-eci:gpu-20261112`。
- Image ID：`sha256:0141b734604ab406f7775009050c2e70c984e313e93658b0704e03e5868b8c7a`。这是本地镜像 ID，推送 ACR 后请使用仓库返回的 digest。
- 大小：9,580,011,570 bytes，Docker 显示约 9.58 GB。
- Linux amd64，Python 3.11，PyTorch / torchaudio 2.8.0+cu126。
- 4 个完整模型目录，38 个文件的大小和 SHA-256 均按 `model-lock.json` 验证。
- FunASR 固定到 `be264e9382b4fef735536f3d278ee14514098a1a`，源码压缩包校验后构建 Python wheel。
- 生产镜像中的 `pip check` 通过，MIT 许可文件保留在 FunASR distribution 中。

## 实际硬件与环境

NVIDIA GeForce GTX 1080 8 GB，驱动 560.94。PyTorch CUDA kernel 真实执行通过，二进制支持的架构包含目标 T4 的 `sm_75`。容器以 UID/GID `10001:10001` 运行。

测试容器的 Docker `NetworkMode` 为 **`none`**。模型加载、全部接口推理和服务重启期间均无容器外部网络；HTTP 客户端通过容器内 loopback 访问服务。

## 功能结果

| 检查 | 结果 |
| --- | --- |
| 单元和 HTTP 接口测试 | 25 passed，Python 3.11，非 root 容器，断网运行 |
| 模型校验、ASR/VAD/标点/声纹本地加载 | 通过 |
| 同人不同语句 1:1 核验 | 分数 0.711194，阈值 0.5，接受 |
| 异人 1:1 核验 | 分数 -0.063643，阈值 0.5，拒绝 |
| 1:N 识别 | 不同语句仍匹配到注册的 `alice` |
| 双说话人转写，已知人数设为 2 | 返回文本、时间戳和 2 个说话人 |
| 转写结果关联声纹身份 | 两个中心分别匹配到 `alice`、`bob`，分数 0.899614、0.977781 |
| 自动估计双说话人数，未传 `speaker_count` | 返回 2 人 |
| 指定人数后再次自动估计单人 | 返回 1 人，没有沿用上次的人数 |
| 静音音频 | 转写返回空结果，声纹识别拒绝 |
| 损坏音频 / 错误 token | 分别返回 422 / 401 |
| 重启后的声纹恢复 | 保留两条注册记录，重启后识别仍正确 |
| 默认 Uvicorn 启动命令及健康探针 | `/readyz` 返回 CUDA 就绪，Docker 状态 `healthy` |
| 没有 GPU，仍指定 CUDA | 启动以 `CUDA is unavailable` 失败，没有回退 CPU |
| Compose 配置 | GPU、loopback 端口和持久卷配置解析通过 |
| ECI 创建请求 | 阿里云 SDK 2.1.0 字段校验和序列化往返通过，没有调用创建接口 |

使用镜像内 CAM++ 模型卡提供的公开录音拼成 21.246 秒的双人样例。包含身份关联的一次转写处理时间为 2.547 秒；默认服务自动估计人数的一次处理时间为 2.037 秒。这是短样例的功能检查，未评测长会议准确率、并发吞吐或 ECI T4 实机性能。

## 复现

```bash
docker build --platform linux/amd64 --target test -t funasr-eci:test .
docker run --rm --network none funasr-eci:test

docker run --rm --gpus all --network none \
  --entrypoint python funasr-eci:gpu scripts/offline_smoke.py
```

验收脚本使用公开的不同语句进行正样本核验，使用另一位说话人的录音进行负样本核验；双人样例使用已知人数 2，另对后续自动单人估计检查参数复位。默认启动容器另外检查了未指定人数的双人样例及 Docker 健康状态。

工作目录 `artifacts/` 保留完整构建日志、接口测试日志、`offline-report.json`、`default-acceptance.json` 和对应容器状态。精简结果同时保存于 [validation-summary.json](validation-summary.json)。

## ECI 部署边界

交付包含可用于 ACR 的本地 GPU 镜像、直接 ECI 请求生成/创建脚本和 ACK 虚拟节点 Pod 示例。尚未推送 ACR，尚未创建 ECI 实例；目标地域的 GPU 库存、配额、NAS 权限及 VPC 镜像访问需要在实际部署环境中确认。驱动配置依据 [ECI 官方 GPU 规格文档](https://help.aliyun.com/document_detail/2579486.html)的 2026 年 11 月版本，模板使用 `tesla=570`。
