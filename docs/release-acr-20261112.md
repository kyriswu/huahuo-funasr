# ACR 发布记录

状态：2026-11-12 已推送成功，并完成远端 manifest 和镜像配置核验。

目标仓库：`huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci`。

版本标签：`gpu-20261112`；同内容标签：`gpu`。

## 拉取

```bash
docker pull huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci:gpu-20261112
```

## ECI 固定镜像地址

```text
huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci@sha256:1434d95d42806a9e748b7bd7dd2272740e8963d0d0823f0c089edea39bc68f60
```

这个地址已填写到 ACK/ECI [Pod 示例](../deploy/ack-eci-pod.yaml)；直接 ECI 请求可将该地址设为 `ECI_IMAGE`。

## 内容核验

两个标签的远端 manifest digest 一致：

```text
sha256:1434d95d42806a9e748b7bd7dd2272740e8963d0d0823f0c089edea39bc68f60
```

发布来源是已通过 [构建与断网 GPU 验收](validation.md) 的本地镜像：

```text
sha256:0141b734604ab406f7775009050c2e70c984e313e93658b0704e03e5868b8c7a
```

远端 `config.digest` 与上述本地 Image ID 完全一致，远端 manifest 原始内容的 SHA-256 与 ACR 返回的 digest 一致；使用固定 digest 的 `docker pull` 也已成功。发布的是同一份已验收镜像。

镜像为 Linux amd64，内置 ASR、VAD、标点及 CAM++ 模型；应用代码来自本地提交 `f4f6861`。共 14 个镜像层，远端压缩层合计 5,921,990,232 bytes，本地解压后约 9.58 GB。

完整上传日志和远端检查结果保存在：

- `artifacts/acr-push-20261112.log`
- `artifacts/acr-push-gpu.log`
- `artifacts/acr-inspect-20261112.txt`
- `artifacts/acr-inspect-gpu.txt`
- `artifacts/acr-manifest-20261112.json`
- `artifacts/acr-verification.json`
- `artifacts/acr-pull-verified.log`

本次仅发布容器镜像，没有创建 ECI 实例。首次从新环境拉取需要配置该 ACR 仓库的访问凭据；运行所需的全部语音模型已内置。
