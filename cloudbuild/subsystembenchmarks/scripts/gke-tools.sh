#!/usr/bin/env bash
set -euo pipefail

# Pinned Helm 3 release verified by SHA-256 checksum (maintained manually;
# macrobenchmarks uses Helm v4 via .github/workflows/bump-helm.yml).
# Digest source: https://get.helm.sh/helm-v3.20.1-linux-amd64.tar.gz.sha256sum
HELM_VERSION="v3.20.1"
HELM_SHA256="0165ee4a2db012cc657381001e593e981f42aa5707acdd50658326790c9d0dc3"

if [[ -n "${SSB_TOOLS_DIR:-}" ]]; then
  TOOLS_DIR="${SSB_TOOLS_DIR}"
elif [[ -d /workspace && -w /workspace ]]; then
  TOOLS_DIR="/workspace/.ssb-tools"
else
  TOOLS_DIR="/usr/local/bin"
fi

install_google_cloud_tools() {
  if command -v gcloud >/dev/null 2>&1 \
    && command -v kubectl >/dev/null 2>&1 \
    && command -v gke-gcloud-auth-plugin >/dev/null 2>&1; then
    return 0
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    apt-transport-https ca-certificates curl git gnupg >/dev/null

  install -m 0755 -d /usr/share/keyrings
  if [[ ! -f /usr/share/keyrings/cloud.google.gpg ]]; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  fi
  cat > /etc/apt/sources.list.d/google-cloud-sdk.list <<'EOF'
deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main
EOF
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    google-cloud-cli kubectl google-cloud-cli-gke-gcloud-auth-plugin >/dev/null
}

install_pinned_helm() {
  if [[ -x "${TOOLS_DIR}/helm" ]] \
    && [[ "$("${TOOLS_DIR}/helm" version --template '{{.Version}}' 2>/dev/null || true)" == "${HELM_VERSION}" ]]; then
    export PATH="${TOOLS_DIR}:${PATH}"
    return 0
  fi

  local tarball="helm-${HELM_VERSION}-linux-amd64.tar.gz"
  local tmp_dir
  tmp_dir=$(mktemp -d)
  if ! curl -fsSL --retry 3 -o "${tmp_dir}/${tarball}" "https://get.helm.sh/${tarball}"; then
    rm -rf "${tmp_dir}"
    echo "ERROR: failed to download ${tarball}" >&2
    return 1
  fi
  if ! echo "${HELM_SHA256}  ${tmp_dir}/${tarball}" | sha256sum -c --quiet - >/dev/null 2>&1; then
    rm -rf "${tmp_dir}"
    echo "ERROR: ${tarball} does not match pinned SHA-256 checksum" >&2
    return 1
  fi
  if ! tar -xzf "${tmp_dir}/${tarball}" -C "${tmp_dir}" linux-amd64/helm; then
    rm -rf "${tmp_dir}"
    return 1
  fi

  mkdir -p "${TOOLS_DIR}"
  install -m 0755 "${tmp_dir}/linux-amd64/helm" "${TOOLS_DIR}/helm"
  rm -rf "${tmp_dir}"

  if [[ "${TOOLS_DIR}" != "/usr/local/bin" && -d /usr/local/bin && -w /usr/local/bin ]]; then
    ln -sf "${TOOLS_DIR}/helm" /usr/local/bin/helm
  fi
  export PATH="${TOOLS_DIR}:${PATH}"
}

install_google_cloud_tools
install_pinned_helm
