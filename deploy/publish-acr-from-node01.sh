#!/usr/bin/env bash
# Install a reviewed copy on Node01 and invoke it as a fixed command.
set -euo pipefail
umask 077

fail() {
  printf 'publish-funasr: %s\n' "$*" >&2
  exit 1
}

if [[ $# != 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  printf '%s\n' 'Usage: publish-acr-from-node01.sh MAIN_COMMIT_SHA' >&2
  exit 2
fi

sha=$1
source_url=https://github.com/kyriswu/huahuo-funasr.git
registry=huahuoai-docker-hub-registry-vpc.cn-beijing.cr.aliyuncs.com
repo_path=${HUAHUO_FUNASR_ACR_REPOSITORY:?Set the authorized ACR namespace/repository on Node01}
[[ "$repo_path" =~ ^[a-z0-9][a-z0-9._-]{0,118}[a-z0-9]/[a-z0-9][a-z0-9._-]{0,62}[a-z0-9]$ ]] ||
  fail 'HUAHUO_FUNASR_ACR_REPOSITORY must be a lowercase namespace/repository without a registry or tag'
repository=$registry/$repo_path
builder=${HUAHUO_FUNASR_BUILDER:-huahuo-funasr}
# The wrapper can supply private Docker credentials without hiding the host's builder.
export BUILDX_CONFIG="${BUILDX_CONFIG:-$HOME/.docker/buildx}"

for dependency in git docker flock; do
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

# Include the private temporary directory's suffix to avoid reusing an attempt's tag.
image=$repository:a10-2gb-$sha-${workdir##*.}
git -C "$workdir" archive --format=tar "$sha" |
  docker buildx build --builder "$builder" --progress=plain --platform linux/amd64 \
    --build-arg BUILD_PROFILE=a10-2gb --target ask-release \
    --label "org.opencontainers.image.revision=$sha" --load --tag "$image" -

verify_main_tip
docker push "$image"
digest=$(docker buildx imagetools inspect --format '{{.Manifest.Digest}}' "$image")
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'ACR did not return a canonical image digest'
printf 'FUNASR_IMAGE=%s@%s\n' "$repository" "$digest" >&3
