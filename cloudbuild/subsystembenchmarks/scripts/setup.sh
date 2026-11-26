#!/usr/bin/env bash
set -euo pipefail

source "$HOME/ssb_cloudbuild.env"
cd "$HOME/gcsfs"

sudo apt-get update >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-pip python3-venv git >/dev/null
python3 -m venv env
source env/bin/activate
pip install --upgrade pip >/dev/null

# Manual gcloud builds submit omits .git, which leaves hatch-vcs without a version.
if [[ ! -d .git ]]; then
  export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"
fi
# shellcheck disable=SC1091
source cloudbuild/subsystembenchmarks/scripts/install-group.sh
install_group_requirements
install_accelerator_packages
# Install local editable gcsfs after group requirements so its fsspec floor wins.
install_editable_gcsfs
install_requirements_override "${REQUIREMENTS_OVERRIDE:-}"

REQUIREMENTS_RESOLVED=$(pip list --format=json)
{
  printf 'export GCSFS_SUBSYSTEM_REQUIREMENTS_OVERRIDE=%q\n' "$GCSFS_SUBSYSTEM_REQUIREMENTS_OVERRIDE"
  printf 'export GCSFS_SUBSYSTEM_REQUIREMENTS_RESOLVED=%q\n' "$REQUIREMENTS_RESOLVED"
} >> "$HOME/ssb_cloudbuild.env"
