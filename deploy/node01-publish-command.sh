#!/usr/bin/env bash
# Cloud Assistant supplies its own invocation and command IDs.
set -euo pipefail
umask 077

if [[ $# != 2 || ! "$1" =~ ^t-[a-zA-Z0-9]+$ || ! "$2" =~ ^c-[a-zA-Z0-9]+$ ]]; then
  printf 'Invalid Cloud Assistant invocation or command ID\n' >&2
  exit 2
fi

invoke_id=$1
command_id=$2
region=cn-beijing
node=i-2zeaydxol26pp3dxg1ak
registry=huahuoai-docker-hub-registry-vpc.cn-beijing.cr.aliyuncs.com
acr_id=cri-mh4e8xf8s1418j8u
ram_role=KubernetesWorkerRole-7991b0e7-f7be-4fdc-8a76-b7de3afb2468
publish_dir=/opt/huahuo-funasr-publish
aliyun=$publish_dir/bin/aliyun

metadata=http://100.100.100.200/latest
token=$(curl -fsS --noproxy '*' --connect-timeout 2 --max-time 5 -X PUT \
  -H 'X-aliyun-ecs-metadata-token-ttl-seconds: 60' "$metadata/api/token")
credentials=$(curl -fsS --noproxy '*' --connect-timeout 2 --max-time 5 \
  -H "X-aliyun-ecs-metadata-token: $token" "$metadata/meta-data/ram/security-credentials/$ram_role")
export ALIBABA_CLOUD_IGNORE_PROFILE=TRUE
ALIBABA_CLOUD_ACCESS_KEY_ID=$(jq -er '.AccessKeyId | select(length > 0)' <<< "$credentials")
ALIBABA_CLOUD_ACCESS_KEY_SECRET=$(jq -er '.AccessKeySecret | select(length > 0)' <<< "$credentials")
ALIBABA_CLOUD_SECURITY_TOKEN=$(jq -er '.SecurityToken | select(length > 0)' <<< "$credentials")
export ALIBABA_CLOUD_ACCESS_KEY_ID ALIBABA_CLOUD_ACCESS_KEY_SECRET ALIBABA_CLOUD_SECURITY_TOKEN
unset token credentials

invocation=$("$aliyun" ecs DescribeInvocations --region "$region" --RegionId "$region" \
  --InstanceId "$node" --InvokeId "$invoke_id" --CommandId "$command_id" \
  --RepeatMode Once --IncludeOutput false --MaxResults 2 --retry-count 0 --yes)
sha=$(jq -er --arg invoke "$invoke_id" --arg command "$command_id" --arg node "$node" '
  .Invocations.Invocation | select(length == 1) | first
  | select(.InvokeId == $invoke and .CommandId == $command and .RepeatMode == "Once")
  | select([.InvokeInstances.InvokeInstance[].InstanceId] == [$node])
  | [.Tags.Tag[] | select(.TagKey == "funasr-source-sha") | .TagValue]
  | select(length == 1) | first | select(test("^[0-9a-f]{40}$"))
' <<< "$invocation")

auth=$("$aliyun" cr GetAuthorizationToken --region "$region" --InstanceId "$acr_id" --ExpiresInHours 2)
jq -e '.IsSuccess == true and .ExpireTime > (now * 1000)' <<< "$auth" >/dev/null
username=$(jq -er '.TempUsername | select(length > 0)' <<< "$auth")
unset ALIBABA_CLOUD_ACCESS_KEY_ID ALIBABA_CLOUD_ACCESS_KEY_SECRET ALIBABA_CLOUD_SECURITY_TOKEN

DOCKER_CONFIG=$(mktemp -d)
export DOCKER_CONFIG
trap 'rm -rf -- "$DOCKER_CONFIG"' EXIT
jq -n --arg host "$registry" '{auths: {($host): {}}}' > "$DOCKER_CONFIG/config.json"
jq -jer '.AuthorizationToken | select(length > 0)' <<< "$auth" |
  docker login "$registry" --username "$username" --password-stdin >/dev/null 2>&1
unset auth username

export BUILDX_CONFIG=/root/.docker/buildx
export HUAHUO_FUNASR_ACR_REPOSITORY=huahuo/huahuo-funasr
export HUAHUO_FUNASR_ACR_STATE_DIR=$publish_dir/state
bash "$publish_dir/publish-acr-from-node01.sh" "$sha"
