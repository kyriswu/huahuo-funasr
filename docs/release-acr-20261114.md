# 异步回调与 ACK 版 ACR 发布记录

日期：2026-11-14。服务代码提交 `043c12c`，接口版本 1.1.0。

仓库：`huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci`。

版本标签：`gpu-async-ack-20261114`；同内容别名：`gpu`。

## 部署地址

```bash
docker pull huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci:gpu-async-ack-20261114
```

固定 digest：

```text
huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci@sha256:dc3ace750eddecdadae29a0be5cd1364f0591e18501cf90c33227eab2627635c
```

本地已验收 Image ID：

```text
sha256:e3483113f71df0dac9b3fc623c4854f71976a9468e6463661663ea52144294dc
```

## 变更

- `POST /v1/tasks`：内存异步转写，支持回调和 Idempotency-Key。
- 独立状态与完整结果 GET，均不消费；`POST /v1/tasks/{id}/ack` 以 event_id 幂等确认消费。
- 带 HMAC 签名的完整结果回调、退避重试；回调 2xx 与主动 ACK 任一成功即清理。
- 解析后立即删除临时音频，未领取结果无自动过期；待领取计入容量，满额 429。
- 任务及结果不持久化，Docker `/tmp` tmpfs 重启清空；注册声纹继续挂载 `/data` 持久化。

镜像继承初版固定 digest 的全部 14 个文件系统层，仅新增 4 个小层。CUDA Runtime 12.6、cuBLAS/cuDNN、PyTorch 2.8.0+cu126、完整模型与 FP32 配置保持原发布内容。解压大小 9,580,097,587 bytes，远端压缩层合计 5,922,016,696 bytes。

## 核验

远端 manifest 原始字节的 SHA-256 与 registry digest 一致，远端 config.digest 与已验收本地 Image ID 一致。固定 digest 拉取成功，版本标签与 gpu 别名指向同一 digest。初版 gpu-20261112 继续指向原 digest。

47 项断网非 root 容器检查通过；真实 GPU 双人转写、声纹、完整回调签名/重试、ACK、默认 CMD healthy 和实际 Docker restart/tmpfs 清空均通过。实际硬件为 GTX 1080 8 GB，未连接 A10 线上服务器，未创建云资源。详见 [验收记录](validation-async.md)。

完整启动、回调验签及 ACK 示例见 [A10 ECS Docker 教程](ecs-a10-async.md)。

本地保留 `artifacts/acr-push-async-20261114.log`、`acr-push-gpu-async-20261114.log`、`acr-manifest-async-20261114.json`、`acr-inspect-async-20261114.json`、`acr-verification-async-20261114.json` 和固定 digest 拉取日志。
