# 异步回调与 ACK 验收

日期：2026-11-14。目标为单机 A10 ECS Docker；验收硬件实际为 GTX 1080 8 GB、驱动 560.94，未连接或修改线上 ECS，未创建云资源。

## 镜像与模型

生产镜像：`funasr-eci:gpu-async-ack-20261114`。

Image ID：`sha256:e3483113f71df0dac9b3fc623c4854f71976a9468e6463661663ea52144294dc`。

Linux amd64，UID/GID 10001:10001，大小 9,580,097,587 bytes。`Dockerfile.release` 复用原发布镜像的全部 14 个文件系统层，增加 4 个小层。构建时模型清单与原镜像逐字节匹配，当前 constraints.txt 的全部 Python 包版本与原镜像匹配。模型、FP32 配置及内置 PyTorch 2.8.0+cu126 / CUDA Runtime 12.6 未变。

完整源码 Dockerfile 也已构建 test 目标。构建机的 Docker Hub 镜像加速源对外部 Dockerfile frontend 返回 403，因此完整构建临时移除 syntax 注释，使用 Docker Engine 内置 frontend；应用与构建步骤未改变。发布 Dockerfile 不需要外部 frontend，构建正常。

## 自动化结果

| 检查 | 结果 |
| --- | --- |
| 非 root、断网测试容器 | 47 passed，7.16 秒；第三方 anyio/Starlette 有一条弃用警告 |
| 状态与结果分离 | 状态查询不含结果，不触发消费 |
| 主动领取与 ACK | 可重复获取相同事件，错误事件拒绝，正确 ACK 清理，重复 ACK 幂等 |
| 回调传输与签名 | 真实 HTTP 发送正文，独立计算 HMAC 匹配，禁止跟随重定向 |
| 重试与并发 | 同一事件/正文重试，失败不重新推理，ACK 与在途回调并发只释放一次容量 |
| 回调耗尽 | 结果继续保留，支持主动获取后 ACK |
| 容量 | 待领取计入上限，满额 429，ACK 后释放；音频总量、单文件、单事件大小上限生效 |
| 无结果自动过期 | 人为老化一天的未领取结果仍可读，仍占用容量 |
| 幂等提交 | 满额下重复 key 仍定位原任务，不同音频冲突；已消费 key 不重新执行 |
| 输入与失败 | 非法音频生成可领取失败事件，音频立即删除 |
| 重启 | 任务不恢复，已消费状态有数量上限，未消费结果不受已消费历史淘汰影响 |

## 真实 GPU 与最终产物

使用镜像内模型卡提供的公开录音，断网运行真实 ASR、双人分段、声纹注册/核验/识别及身份关联。结果通过；同人/异人核验及身份分数与原版本一致。异步转写返回带时间戳的两个人物分段，解析后立即删除音频，重复 GET 不消费，ACK 后返回 410。

容器内 HTTP 回调接收端独立验签，第一次返回 503，第二次保存事件后返回 204；两次事件 ID 与原始正文相同。服务清理结果，且只执行一次推理。服务进程重启后未领取结果变为 404，声纹库仍在。

最终生产镜像使用默认 Uvicorn CMD 再次验收，未挂载源代码；GPU、`--network none`、`--tmpfs /tmp`。Docker health 为 healthy，真实异步双人转写、主动保存后 ACK、真实签名回调重试与自动清理均通过。实际执行 `docker restart` 后，旧任务 404，事先写入 tmpfs 的标记文件消失，模型重新就绪且 health 恢复 healthy。

样例录音约 21 秒，耗时受构建和测试时宿主机负载影响；这些记录用于功能验收，不作为 A10 性能或“零开销”的基准。A10 二进制架构 sm_86 已在基础镜像确认，A10 线上硬件仍需部署后验收。

## 日志

- `artifacts/unit-async-20261114.log`
- `artifacts/build-test-async-20261114.log`
- `artifacts/build-test-async-release-20261114.log`
- `artifacts/build-async-20261114.log`
- `artifacts/offline-async-20261114.log`
- `artifacts/async-validation/offline-report.json`
- `artifacts/default-async-report.json`
- `artifacts/default-async-restart-report.json`

部署示例和回调契约见 [A10 ECS Docker 教程](ecs-a10-async.md)。
