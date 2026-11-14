# A10 ECS Docker 部署与异步回调

此版本在现有完整模型、FP32 推理与 PyTorch 2.8.0+cu126 上增加异步任务、回调和 ACK。镜像自带 CUDA Runtime 12.6、cuBLAS 和 cuDNN。ECS 需要 NVIDIA 驱动和 NVIDIA Container Toolkit，不需安装宿主机 CUDA Toolkit；支持 CUDA 12.8 的驱动可兼容运行。

发布使用 `Dockerfile.release` 继承已验收的固定 digest，只增加服务代码和脚本层，构建时校验模型清单及 Python 包版本与当前约束一致。完整源码构建继续使用 `Dockerfile`；修改模型或依赖时需要完整构建。

## 启动

以下命令在目标 ECS 执行。先确认 `nvidia-smi` 可以看到 A10，Docker 已配置 GPU 支持。镜像拉取约 6 GB，解压后约 9.6 GB，预留足够磁盘。

```bash
docker login huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com
export IMAGE=huahuo-docker-hub-registry.cn-beijing.cr.aliyuncs.com/ai_service/funasr-eci:gpu-async-ack-20261114
docker pull "$IMAGE"

# 首次创建；保存此文件，不要在每次重启时重生成 token。
umask 077
cat > funasr.env <<EOF
DEVICE=cuda:0
API_TOKEN=$(openssl rand -hex 32)
CALLBACK_SECRET=$(openssl rand -hex 32)
CALLBACK_ALLOWED_ORIGINS=https://business.example.com
MAX_PENDING_TASKS=16
MAX_TASK_AUDIO_MB=1024
MAX_TASK_RESULT_MB=16
EOF
# 将 CALLBACK_ALLOWED_ORIGINS 改成你的业务服务来源。
# VPC 内回调可显式配置 http://10.0.0.8:8080，不含回调路径。

docker run -d --name funasr --gpus all --restart unless-stopped \
  --stop-timeout 600 \
  -p 8000:8000 \
  --env-file funasr.env \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 \
  --mount type=volume,src=funasr-data,dst=/data \
  --log-opt max-size=20m --log-opt max-file=3 \
  "$IMAGE"

docker logs --tail 50 funasr
curl --fail http://127.0.0.1:8000/readyz
```

首次启动需要校验及加载模型。保留默认单进程启动命令。安全组按业务调用来源开放 8000；对公网提供服务时接入现有 HTTPS 网关。Swagger 地址为 `http://ECS地址:8000/docs`，Authorize 填 `funasr.env` 中的 API_TOKEN。下游单独配置同一 CALLBACK_SECRET。

`/tmp` 使用 tmpfs，设置 2 GiB 上限，实际使用会计入宿主机内存。原始音频任务池另有 1 GiB 上限，为上传暂存、解码和缓存留出余量。声纹库 `/data/voiceprints` 使用持久卷。不要把任务临时目录挂到 NAS。

## 提交与正常回调

```bash
export BASE_URL=http://127.0.0.1:8000
# 从 funasr.env 取出 token，避免打印到终端。
export API_TOKEN="$(sed -n 's/^API_TOKEN=//p' funasr.env)"

curl --fail-with-body "$BASE_URL/v1/tasks" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H 'Idempotency-Key: meeting-20261114-001' \
  -F 'file=@meeting.mp3' \
  -F 'callback_url=https://business.example.com/asr/completed' \
  -F 'identify_speakers=true'
```

上传完整保存到 tmpfs 后返回 202，例如 `{"task_id":"task_...","status":"queued",...}`。未完成音频解码或推理不会阻塞提交响应，文件上传自身仍需足够的客户端/网关超时。已知说话人数可加 `-F speaker_count=2`，声纹身份须先注册。无需回调时省略 callback_url，通过结果接口与 ACK 领取。

同一 Idempotency-Key、相同音频和参数在当前进程保留期内返回同一任务；换音频或参数返回 409。回调已经消费的任务再次提交相同 key 不会重新推理，返回其已消费状态。已消费简要状态过期、被容量淘汰或服务重启后，不再保证此 key 去重，应使用新的业务键管理重提。

完成后服务向 callback_url POST 完整事件，正文结构如下：

```json
{
  "event_id": "evt_0123456789abcdef0123456789abcdef",
  "event": "task.succeeded",
  "task_id": "task_0123456789abcdef0123456789abcdef",
  "status": "succeeded",
  "result": {"text": "转写内容", "segments": [], "speakers": []},
  "error": null
}
```

`result` 完整字段与 `/v1/transcriptions` 一致，包含文本、毫秒时间戳、说话人及可选声纹身份。失败事件为 `task.failed`、`result=null`，`error` 提供 code 和 message；失败结果也需要消费。

下游先验签、保存事件和业务结果，再返回任意 2xx。回调 2xx 即确认消费，服务自动清空结果并释放容量。不要在业务数据未保存时提前返回成功。签名或业务保存失败则返回非 2xx，服务会重试；此时不需再发 ACK。

## 回调验签与去重

请求头包含：

```text
X-FunASR-Event-Id: evt_...
X-FunASR-Timestamp: Unix时间戳（秒）
X-FunASR-Signature: v1=<十六进制HMAC-SHA256>
```

签名输入是 `timestamp + "." + event_id + "." + 原始请求体字节`。不要先解析再序列化正文参与验签。Python 验签核心如下：

```python
import hashlib
import hmac
import time

def valid_callback(secret, headers, raw_body):
    stamp = headers.get("X-FunASR-Timestamp", "")
    event_id = headers.get("X-FunASR-Event-Id", "")
    try:
        if abs(time.time() - int(stamp)) > 300:
            return False
        signed = stamp.encode("ascii") + b"." + event_id.encode("ascii") + b"." + raw_body
    except (ValueError, UnicodeError):
        return False
    expected = "v1=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(headers.get("X-FunASR-Signature", ""), expected)
```

验签后校验正文 event_id 与请求头一致，以 event_id 做业务幂等。重试保持事件 ID 和原始正文不变，每次更新签名时间戳。已保存的重复事件直接返回 2xx。回调与主动 ACK 并发时，已经发出的通知仍可能到达，仍需去重。

默认回调读写超时 10 秒，非 2xx/网络异常均退避重试，总共最多 10 次；重试间隔从 10 秒指数增长，最高一小时并带抖动。3xx 不跟随。重试耗尽后 callback.status 为 failed，结果仍可通过查询接口取回并 ACK，不会重新转写。

## 查询兜底与 ACK

```bash
export TASK_ID=task_替换为提交返回的ID

# 只查状态，不含转写内容，不清理结果。
curl --fail-with-body "$BASE_URL/v1/tasks/$TASK_ID" \
  -H "Authorization: Bearer $API_TOKEN"

# 任务终态后获取完整事件。GET 可重复读取，不会清理。
curl --fail-with-body "$BASE_URL/v1/tasks/$TASK_ID/result" \
  -H "Authorization: Bearer $API_TOKEN" \
  --output result.json

# 确认 result.json 已保存/业务已入库后再执行以下 ACK。
python3 -c 'import json; print(json.dumps({"event_id": json.load(open("result.json"))["event_id"]}))' \
  | curl --fail-with-body "$BASE_URL/v1/tasks/$TASK_ID/ack" \
      -H "Authorization: Bearer $API_TOKEN" \
      -H 'Content-Type: application/json' --data-binary @-
```

ACK 响应丢失时，可以用相同 event_id 再次确认，幂等返回成功。错误 event_id 返回 409，不会清理。结果接口未完成为 409，已消费为 410，任务未知/重启前为 404。

## 清理与容量

- 解析成功或失败后立即删除上传音频及解码文件，不等待结果领取。
- 回调 2xx 或匹配 event_id 的显式 ACK 清空结果，停止未来回调，释放任务容量。
- 未领取结果不按五分钟或其他时间自动删除。回调耗尽也继续保留，直到 ACK 或进程重启。
- 默认最多 16 个未消费任务，包含上传中、排队、运行中和待领取结果；满额返回 429 和 Retry-After: 2。查询结果不释放容量，ACK 才释放。
- 单个事件默认最多 16 MiB，超过后返回小型失败事件，避免待领取内存无限增长。16 个任务的事件正文总量上限约 256 MiB，另有执行和网络发送时的临时内存。
- 已消费任务只保留简要状态，默认一小时、最多 1024 条，到期或超量淘汰。未消费任务不受此期限影响。
- 进程/容器重启后丢弃任务与结果，无恢复；注册声纹仍保存在 Docker 持久卷。

正常业务使用回调，不需轮询。回调不可用或耗尽时查询并 ACK。监控 429、callback.failed 和待领取数量，发现业务接收故障后优先恢复下游或主动领取。
