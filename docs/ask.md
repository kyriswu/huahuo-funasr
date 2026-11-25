# ACS NVIDIA GPU 接入与 ASK ECI 验证

## 镜像发布

主分支自动发布只构建并推送 Linux amd64 的 `nvidia-modern`。工作流复用 `huahuo-monitor` 的 `github-actions` Provider 和 `github-actions-release` 角色。测试通过后，工作流调用 Node01 的 FunASR 固定云助手命令，将镜像推送到 `huahuoai-docker-hub-registry-vpc.cn-beijing.cr.aliyuncs.com`。 [GitHub Actions 工作流](../.github/workflows/publish-acr.yml)

- 共享角色需信任 `repo:kyriswu@5120703/huahuo-funasr@1460054211:ref:refs/heads/main`，并允许调用 FunASR 的固定命令及查询其结果。
- Node01 的固定命令 `c-bj0742lf7bciz28` 读取 `funasr-source-sha` 和 `funasr-build-profile` 调用标签，使用专用 Buildx 构建器运行 [发布脚本](../deploy/publish-acr-from-node01.sh)，由实例角色获取临时凭证推送 `huahuo/huahuo-funasr`。 [固定命令入口](../deploy/node01-publish-command.sh)
- GitHub Actions 运行摘要记录可用于部署的 `FUNASR_IMAGE` 固定摘要。CI 继续检查 `a10-2gb` 配置，该版本按需手动构建。
- Node01 使用阿里云镜像站下载构建依赖，并核验 PyTorch 和 TorchAudio wheel 的上游 SHA256。构建时可以复用匹配的 Python OCI 缓存及通过原始 SHA256 校验的 FunASR 源码包。
- 在同一个 BuildKit 构建器上，两种配置共用模型缓存。下载中断后可以续传，权重的大小及 SHA256 匹配模型清单后才会写入镜像。

| 构建配置 | CUDA | 模型与用途 |
| --- | --- | --- |
| `a10-2gb` | 11.8 | 使用[轻量中文标点模型](../model-lock-punc-small.json)，默认 `BATCH_SIZE_S=10`，面向 A10 1/12 的 2 GB 显存优化。实际显存峰值需按下文验收。 |
| `nvidia-modern` | 12.8 | 使用[完整生产模型集](../model-lock.json)，保留大型中英标点模型和 FP32，默认 `BATCH_SIZE_S=120`，用于驱动兼容且显存充足的 GPU。 |

两版均以 `DEVICE=cuda:0` 运行，ASR、VAD、标点和声纹模型都在 GPU 上。CUDA 不可用时启动报错，推理显存耗尽时返回任务错误。

以下命令可手动构建任一配置。A10 1/12 使用 `a10-2gb`。

```bash
docker buildx build --platform linux/amd64 --build-arg BUILD_PROFILE=a10-2gb --target ask-release --load -t funasr-eci:a10-2gb .
docker buildx build --platform linux/amd64 --build-arg BUILD_PROFILE=nvidia-modern --target ask-release --load -t funasr-eci:nvidia-modern .
```

## 内网服务申请 ACS NVIDIA GPU

ACS 整卡优先使用 `nvidia-modern` 的发布摘要，按表填写卡型和驱动。表中的较早架构卡型也可使用这一版，首次接入按下文确认显存峰值和真实转写。需要用 `a10-2gb` 处理其他低显存卡型时，先验证 CUDA 11.8 的驱动、GPU 架构和实际显存需求。Blackwell 卡型使用 `nvidia-modern` 的 CUDA 12.8。 [NVIDIA CUDA 架构矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)

| `alibabacloud.com/gpu-model-series` | 架构 | `alibabacloud.com/gpu-driver-version` | 阿里云公开列出的北京可用区 |
| --- | --- | --- | --- |
| `T4` | Turing | `580.95.05-open` | 未列出 |
| `A10` | Ampere | `580.95.05-open` | `cn-beijing-i` |
| `GN8IS`（L20） | Ada | `580.95.05-open` | `cn-beijing-l` |
| `G49E` | Ada | `580.95.05-open` | `cn-beijing-l` |
| `GU8TEF` | Hopper | `580.95.05-open` | `cn-beijing-d` |
| `GU8TF` | 待实机确认 | `580.95.05-open` | `cn-beijing-d` |
| `G28Ti` | 待实机确认 | `580.95.05-open` | `cn-beijing-h`、`cn-beijing-i` |
| `G59` | Blackwell | `580.95.05-open` | `cn-beijing-l` |
| `L20N` | Blackwell | `580.95.05-open` | `cn-beijing-i` |
| `L20NE` | Blackwell | `580.126.09-open` | `cn-beijing-l` |

表中是普通按量单卡候选。申请服务按现有地域、交换机和账号可用的实时库存选择，每个 Pod 填一个卡型。`GX8SF`（L20X）的普通按量规格为 8 卡，公开目录只列新加坡。`P16EN`、`PPU810E` 和 `PPU890P` 属于 PPU 系列。 [ACS 卡型和可用区](https://help.aliyun.com/en/cs/user-guide/gpu/) · [ACS GPU 规格](https://help.aliyun.com/en/cs/user-guide/gpu-families-supported-by-acs) · [ACS 驱动表](https://help.aliyun.com/zh/cs/user-guide/list-of-gpu-driver-versions-supported-by-acs/) · [ACK `GN8IS` 示例](https://help.aliyun.com/zh/cs/user-guide/access-acs-computing-power-in-an-ack-cluster) · [阿里云卡型架构](https://help.aliyun.com/en/functioncompute/some-gpu-card-types-do-not-provide-sla-commitment-statement) · [GU8TEF 硬件映射](https://help.aliyun.com/zh/rds/apsaradb-rds-for-mysql/scheduled-scaling-for-rds-custom-ai-nodes) · [NVIDIA Hopper H20](https://docs.nvidia.com/ai-enterprise/release-9/latest/infra-software/vgpu/reference/hopper-h20.html)

在 Pod 标签中设置 `alibabacloud.com/compute-class: gpu`、表中的 `gpu-model-series` 和 `gpu-driver-version`。容器的 `requests` 和 `limits` 均设置为 `cpu: "4"`、`memory: 16Gi` 和 `nvidia.com/gpu: "1"`。ACS 按 1 GiB 步进，16 GiB 覆盖原来的 15.5 GiB。现有策略若使用 ACS BestEffort 算力质量，标签为 `alibabacloud.com/compute-qos: best-effort`，可能发生抢占。`default` 也是 ACS GPU 支持的算力质量。Kubernetes Pod 的资源配置仍按前述 `requests` 和 `limits` 设置。 [ACS GPU 标签](https://help.aliyun.com/zh/cs/user-guide/pod-gpu-model-and-driver-version) · [ACS 算力质量](https://help.aliyun.com/zh/cs/user-guide/acs-pod-instance-overview) · [ACK Serverless 接入 ACS](https://help.aliyun.com/zh/cs/user-guide/use-acs-computing-power-through-the-ack-serverless-cluster)

首次使用每种卡型时，记录 `nvidia-smi` 显示的实际型号、驱动和显存，确认 `/readyz` 返回 `cuda:0`，再用具有代表性的公开样例录音完成一次真实转写，检查 CUDA 显存峰值。`GU8TF` 和 `G28Ti` 首次接入时，也在容器内检查实际计算能力及镜像内的 CUDA 架构列表，通过后再纳入常用候选。

```bash
python -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())'
```

## ECI 2 GB 验证

[验证清单](../deploy/ask-smoke.yaml)为初始副本数为 0 的单副本 Deployment，使用 ECI 专有的 `k8s.aliyun.com/eci-use-specs` 请求 `ecs.sgn7i-vws-m2.xlarge` 的 A10 1/12。该档默认驱动为 GRID 470 / CUDA 11.4，使用 `a10-2gb` 镜像和 `BATCH_SIZE_S=10` 验收 CUDA 启动、转写和显存峰值。 [ASK ECI GPU 规格](https://help.aliyun.com/zh/ack/serverless-kubernetes/user-guide/create-an-eci-pod)

将 `FUNASR_IMAGE` 设置为已有的 `a10-2gb` 镜像摘要，或手动构建后推送到验证集群可访问的仓库并使用其摘要。

选择已有的 ASK 验证命名空间和镜像访问权限。复用该命名空间的镜像拉取配置，必要时在 Pod 模板中引用已有的 `imagePullSecrets`。清单从同命名空间的 `huahuo-funasr-ask-smoke` Secret 的 `token` 键读取测试 API token。声纹和任务数据只写入该验证 Pod。

```bash
export ASK_NAMESPACE=ns-applications
export FUNASR_IMAGE=registry.example.invalid/huahuo-funasr:a10-2gb

kubectl config current-context
kubectl -n "$ASK_NAMESPACE" apply -f deploy/ask-smoke.yaml
kubectl -n "$ASK_NAMESPACE" set image deployment/huahuo-funasr-ask-smoke api="$FUNASR_IMAGE"
kubectl -n "$ASK_NAMESPACE" scale deployment/huahuo-funasr-ask-smoke --replicas=1
kubectl -n "$ASK_NAMESPACE" rollout status deployment/huahuo-funasr-ask-smoke --timeout=15m
```

在 Pod 就绪后确认实际规格、驱动和显存，再通过本地端口访问 API 和指标。

```bash
kubectl -n "$ASK_NAMESPACE" get pods -l app.kubernetes.io/instance=huahuo-funasr-ask-smoke \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.k8s\.aliyun\.com/eci-instance-spec}{"\n"}{end}'
kubectl -n "$ASK_NAMESPACE" exec deployment/huahuo-funasr-ask-smoke -c api -- nvidia-smi
kubectl -n "$ASK_NAMESPACE" port-forward deployment/huahuo-funasr-ask-smoke 8000:8000 9464:9464
```

在另一个终端确认 `/readyz` 返回 `cuda:0`，并读取 `http://127.0.0.1:9464/metrics`。用公开样例录音按[异步任务流程](ecs-a10-async.md#提交与正常回调)提交转写。通过[结果查询和 ACK](ecs-a10-async.md#查询兜底与-ack)验证结果已持久化、ACK 成功、再次读取返回 410。如需验证签名回调，可以在验证 Deployment 中配置 `CALLBACK_SECRET` 和 `CALLBACK_ALLOWED_ORIGINS` 并使用测试接收端。

同集群的 `huahuo-monitor` 通过 Pod 标签发现 `api:9464`。用 `up{job="application",service="huahuo-funasr",role="api-server"}` 确认已抓取，再检查 `funasr_ready`、转写各阶段耗时、未消费任务数及 PyTorch CUDA 显存峰值。缩容前等待至少一次成功抓取。跨集群时需另外配置发现目标，同一 VPC 提供网络路径。

业务结果已确认消费后，将验证 Deployment 缩回零副本。

```bash
kubectl -n "$ASK_NAMESPACE" scale deployment/huahuo-funasr-ask-smoke --replicas=0
kubectl -n "$ASK_NAMESPACE" wait --for=delete pod -l app.kubernetes.io/instance=huahuo-funasr-ask-smoke --timeout=11m
kubectl -n "$ASK_NAMESPACE" get pods -l app.kubernetes.io/instance=huahuo-funasr-ask-smoke
```
