# ASK GPU 验证

[验证清单](../deploy/ask-smoke.yaml)为独立的单副本 Deployment，初始副本数为 0。它请求 `ecs.sgn7i-vws-m2.xlarge` 的 A10 1/12，默认驱动为 GRID 470 / CUDA 11.4。候选镜像使用 `cu118` 和轻量标点模型，驱动兼容性和显存峰值需在这一档 Pod 上验收。 [ASK GPU 规格说明](https://help.aliyun.com/zh/ack/serverless-kubernetes/user-guide/create-an-eci-pod)

[GitHub Actions](../.github/workflows/publish-acr.yml)复用 `huahuo-monitor` 的 `github-actions` Provider 和 `github-actions-release` 角色。测试通过后，工作流调用 Node01 上的 FunASR 固定云助手命令。Node01 构建 Linux amd64 镜像，使用已有内网仓库凭据推送到 `huahuoai-docker-hub-registry-vpc.cn-beijing.cr.aliyuncs.com`，再返回固定 digest。

- 共享角色需信任 `repo:kyriswu@5120703/huahuo-funasr@1460054211:ref:refs/heads/main`，并允许调用 FunASR 的固定命令及查询其结果。
- Node01 的固定命令 `c-bj0742lf7bciz28` 读取 `funasr-source-sha` 调用标签，使用专用 Buildx 构建器运行 [发布脚本](../deploy/publish-acr-from-node01.sh)，由实例角色获取临时凭证推送 `huahuo/huahuo-funasr`。 [固定命令入口](../deploy/node01-publish-command.sh)
- GitHub 运行摘要记录可用于部署的 `FUNASR_IMAGE` digest。

也可以在本机构建同一候选镜像。

```bash
docker buildx build --platform linux/amd64 --build-arg BUILD_PROFILE=a10-2gb --load -t funasr-eci:a10-2gb .
```

选择已有的 ASK 验证命名空间和镜像访问权限。复用该命名空间的镜像拉取配置，必要时在 Pod 模板中引用已有的 `imagePullSecrets`。清单从同命名空间的 `huahuo-funasr-ask-smoke` Secret 的 `token` 键读取测试 API token。声纹和任务数据只写入该验证 Pod。

```bash
export ASK_NAMESPACE=ns-applications
export FUNASR_IMAGE=registry.example.invalid/huahuo-funasr:ask-smoke

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
