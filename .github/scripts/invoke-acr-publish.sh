#!/bin/bash
set -euo pipefail

fail() {
    printf '::error::%s\n' "$1" >&2
    exit 1
}

sha=${GITHUB_SHA:?}
run_id=${GITHUB_RUN_ID:?}
run_attempt=${GITHUB_RUN_ATTEMPT:?}
instance=${FUNASR_NODE01_INSTANCE_ID:?}
command=${FUNASR_NODE01_PUBLISH_COMMAND_ID:?}
region=${ACR_REGION:?}
profile=${FUNASR_BUILD_PROFILE:?}
registry=huahuoai-docker-hub-registry-vpc.cn-beijing.cr.aliyuncs.com
repository=${ACR_REPOSITORY:-}
repository_pattern='[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*'

[[ "$sha" =~ ^[0-9a-f]{40}$ && "$run_id" =~ ^[0-9]+$ && "$run_attempt" =~ ^[0-9]+$ &&
   "$instance" == i-2zeaydxol26pp3dxg1ak && "$command" =~ ^c-[a-zA-Z0-9]+$ &&
   "$region" == cn-beijing && "${ACR_HOST:-$registry}" == "$registry" &&
   ( "$profile" == a10-2gb || "$profile" == nvidia-modern ) &&
   ( -z "$repository" || "$repository" =~ ^$repository_pattern$ ) ]] ||
    fail 'Invalid Node01 ACR publish configuration'
: "${ALIBABA_CLOUD_SECURITY_TOKEN:?Temporary Alibaba Cloud credentials are required}"

client_token="funasr-$run_id-$run_attempt-$profile"
# The fixed command reads these tags with its trusted invocation ID.
if ! invoke_id=$(timeout -k 5s 30s aliyun ecs InvokeCommand --region "$region" --RegionId "$region" \
    --InstanceId.1 "$instance" --CommandId "$command" --Username root \
    --Tag.1.Key funasr-source-sha --Tag.1.Value "$sha" \
    --Tag.2.Key funasr-build-profile --Tag.2.Value "$profile" \
    --ClientToken "$client_token" --RepeatMode Once --Timeout 4500 --retry-count 0 --yes \
    2>/dev/null | jq -er '.InvokeId' 2>/dev/null); then
    fail 'Could not start the fixed Node01 ACR publish command'
fi
[[ "$invoke_id" =~ ^t-[a-zA-Z0-9]+$ ]] || fail 'Invalid Cloud Assistant invocation ID'
printf 'Cloud Assistant invocation: %s\n' "$invoke_id" >&2

deadline=$((SECONDS + 4800))
while (( SECONDS < deadline )); do
    if ! result=$(timeout -k 5s 30s aliyun ecs DescribeInvocationResults --region "$region" --RegionId "$region" \
        --CommandId "$command" --InstanceId "$instance" --InvokeId "$invoke_id" \
        --ContentEncoding PlainText --MaxResults 1 --retry-count 0 --yes 2>/dev/null); then
        fail "Could not check Node01 publish invocation $invoke_id"
    fi
    if ! item=$(jq -ce '
        (.Invocation.InvocationResults.InvocationResult // []) as $rows
        | if ($rows | length) > 1 then error("unexpected invocation result")
          else $rows[0] // {} end
    ' <<< "$result" 2>/dev/null); then
        fail "Invalid result for Node01 publish invocation $invoke_id"
    fi
    if [[ "$item" != '{}' ]]; then
        jq -e --arg i "$instance" --arg c "$command" --arg t "$invoke_id" \
            '.InstanceId == $i and .CommandId == $c and .InvokeId == $t' <<< "$item" >/dev/null ||
            fail "Mismatched result for Node01 publish invocation $invoke_id"
        case "$(jq -r '.InvocationStatus' <<< "$item")" in
            Pending|Running|Stopping) ;;
            Success)
                if ! image=$(jq -er --arg profile "$profile" '
                    def receipt($key):
                        map(select(startswith($key)))
                        | select(length == 1) | first | ltrimstr($key);
                    select(.ExitCode == 0 and ((.Dropped // 0) == 0))
                    | .Output | split("\n") | map(sub("\r$"; ""))
                    | select(receipt("FUNASR_BUILD_PROFILE=") == $profile)
                    | receipt("FUNASR_IMAGE=")
                ' <<< "$item" 2>/dev/null); then
                    fail "Invalid image receipt for Node01 publish invocation $invoke_id"
                fi
                reference=${image#"$registry/"}
                [[ "$image" == "$registry/"* && "$reference" =~ ^$repository_pattern@sha256:[0-9a-f]{64}$ &&
                   ( -z "$repository" || "${reference%@*}" == "$repository" ) ]] ||
                    fail "Unexpected image in Node01 publish invocation $invoke_id"
                if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
                    printf 'FUNASR_IMAGE=%s\n' "$image" >> "$GITHUB_OUTPUT"
                fi
                printf 'FUNASR_IMAGE=%s\n' "$image"
                exit 0
                ;;
            *) fail "Node01 publish failed; check Cloud Assistant execution $invoke_id" ;;
        esac
    fi
    sleep 5
done
fail "Timed out waiting for Node01 publish invocation $invoke_id"
