#!/usr/bin/env bash
# Install a reviewed copy on Node01 and invoke it as a fixed command.
set -euo pipefail
umask 077

fail() {
  printf 'publish-funasr: %s\n' "$*" >&2
  exit 1
}

if [[ $# != 2 || ! "$1" =~ ^[0-9a-f]{40}$ || ! "$2" =~ ^(a10-2gb|nvidia-modern)$ ]]; then
  printf '%s\n' 'Usage: publish-acr-from-node01.sh MAIN_COMMIT_SHA a10-2gb|nvidia-modern' >&2
  exit 2
fi

sha=$1
profile=$2
source_url=https://github.com/kyriswu/huahuo-funasr.git
registry=huahuoai-docker-hub-registry-vpc.cn-beijing.cr.aliyuncs.com
repo_path=${HUAHUO_FUNASR_ACR_REPOSITORY:?Set the authorized ACR namespace/repository on Node01}
[[ "$repo_path" =~ ^[a-z0-9][a-z0-9._-]{0,118}[a-z0-9]/[a-z0-9][a-z0-9._-]{0,62}[a-z0-9]$ ]] ||
  fail 'HUAHUO_FUNASR_ACR_REPOSITORY must be a lowercase namespace/repository without a registry or tag'
repository=$registry/$repo_path
builder=${HUAHUO_FUNASR_BUILDER:-huahuo-funasr}
# The wrapper can supply private Docker credentials without hiding the host's builder.
export BUILDX_CONFIG="${BUILDX_CONFIG:-$HOME/.docker/buildx}"

for dependency in git docker flock timeout; do
  command -v "$dependency" >/dev/null || fail "$dependency is required"
done

state_dir=${HUAHUO_FUNASR_ACR_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/huahuo-funasr}
mkdir -p -- "$state_dir"
exec 9>"$state_dir/publish.lock"
flock -n 9 || fail 'another FunASR release is running on Node01'

workdir=$(mktemp -d "$state_dir/publish.$sha.XXXXXX")
log_file=$workdir/publish.log
exec 3>&1 4>&2
finish() {
  local status=$?
  trap - EXIT
  if ((status != 0)); then
    printf 'FunASR publication failed for %s. Last log output:\n' "$sha" >&4
    tail -n 20 -- "$log_file" | tail -c 3000 >&4 || true
  fi
  rm -rf -- "$workdir" || true
  exit "$status"
}
trap finish EXIT
exec >>"$log_file" 2>&1

git -C "$workdir" init -q
verify_main_tip() {
  GIT_TERMINAL_PROMPT=0 git -C "$workdir" fetch -q --no-tags --depth=1 "$source_url" refs/heads/main
  [[ "$(git -C "$workdir" rev-parse --verify 'FETCH_HEAD^{commit}')" == "$sha" ]] ||
    fail 'requested revision is no longer the tip of FunASR main'
}
verify_main_tip

node_build_args=(
  --build-arg DEBIAN_MIRROR=http://mirrors.cloud.aliyuncs.com
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
  --build-arg TORCH_WHEEL_MIRROR=http://mirrors.cloud.aliyuncs.com/pytorch-wheels
)
# Use the verified amd64 base cached on Node01 when it matches this commit's Python pin.
python_pin=$(git -C "$workdir" show "$sha:Dockerfile" |
  sed -nE 's/^ARG PYTHON_IMAGE=.*@sha256:([0-9a-f]{64})$/\1/p')
python_layout=$state_dir/python-oci/$python_pin
if [[ -n "$python_pin" && -r "$python_layout/amd64-digest" ]]; then
  python_digest=$(<"$python_layout/amd64-digest")
  [[ "$python_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'invalid cached amd64 Python digest'
  node_build_args+=(--build-arg PYTHON_IMAGE=funasr-python
    --build-context "funasr-python=oci-layout://$python_layout@$python_digest")
fi

# Include the private temporary directory's suffix to avoid reusing an attempt's tag.
image=$repository:$profile-$sha-${workdir##*.}
deadline=$((SECONDS + 4350))
build_image() {
  local remaining=$((deadline - SECONDS))
  ((remaining > 0)) || fail 'Node01 build deadline exceeded'
  git -C "$workdir" archive --format=tar "$sha" |
    timeout --signal=INT --kill-after=15s "${remaining}s" \
    docker buildx build --builder "$builder" --progress=plain --platform linux/amd64 \
      --build-arg "BUILD_PROFILE=$profile" --target ask-release \
      --label "org.opencontainers.image.revision=$sha" "${node_build_args[@]}" "$@" -
}

build_image --output type=cacheonly

verify_main_tip
build_image --push --tag "$image"
digest=$(docker buildx imagetools inspect --format '{{.Manifest.Digest}}' "$image")
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'ACR did not return a canonical image digest'
printf 'FUNASR_IMAGE=%s@%s\n' "$repository" "$digest" >&3
