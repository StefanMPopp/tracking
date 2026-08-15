#!/usr/bin/env bash
# =============================================================================
# install.sh — installer and updater for the insect tracking pipeline
#
# Fresh machine (download and run):
#   curl -fsSL https://raw.githubusercontent.com/<yourhandle>/tracking/main/install.sh | bash
#
# Or if you downloaded the file manually:
#   bash install.sh
#
# Already installed (update repo + re-provision):
#   bash ~/tracking/install.sh
#   — or equivalently —
#   bash ~/tracking/setup.sh        (setup.sh is a thin alias for this file)
#
# Behaviour:
#   - If the repo already exists  → git pull, then re-run provisioning steps
#   - If the repo does not exist  → git clone, then run provisioning steps
#   Every provisioning step is idempotent: safe to re-run at any time.
#
# Steps:
#   1. git
#   2. Clone or update repo
#   3. Miniforge
#   4. TRex conda environment
#   5. uv
#   6. Python dependencies (uv sync)
#   7. pipeline.yaml (from template, if absent)
# =============================================================================

set -euo pipefail

GITHUB_USER="StefanMPopp"
REPO_NAME="tracking"
INSTALL_DIR="${HOME}/tracking"
PIPELINE_DIR="${INSTALL_DIR}/pipeline"
CONDA_ENV_NAME="trex"
MINIFORGE_DIR="${HOME}/miniforge3"

# =============================================================================
echo ""
echo "============================================================"
echo "  Insect Tracking Pipeline — Install / Update"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. git
# -----------------------------------------------------------------------------
echo ""
echo "--- 1/7  git ---"
if command -v git &>/dev/null; then
    echo "Already installed ($(git --version)) — skipping."
else
    echo "Installing git..."
    sudo apt-get update -qq
    sudo apt-get install -y git
fi

# -----------------------------------------------------------------------------
# 2. Clone or update repo
# -----------------------------------------------------------------------------
echo ""
echo "--- 2/7  Repository ---"
if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "Repo found at ${INSTALL_DIR} — pulling latest changes."
    git -C "${INSTALL_DIR}" pull
    IS_UPDATE=true
else
    echo "Cloning https://github.com/${GITHUB_USER}/${REPO_NAME}.git → ${INSTALL_DIR}"
    git clone "https://github.com/${GITHUB_USER}/${REPO_NAME}.git" "${INSTALL_DIR}"
    IS_UPDATE=false
fi

# -----------------------------------------------------------------------------
# 3. Miniforge
# -----------------------------------------------------------------------------
echo ""
echo "--- 3/7  Miniforge ---"
if [ -d "${MINIFORGE_DIR}" ]; then
    echo "Already installed at ${MINIFORGE_DIR} — skipping."
else
    echo "Downloading Miniforge..."
    INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -o "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${MINIFORGE_DIR}"
    rm "${INSTALLER}"
    "${MINIFORGE_DIR}/bin/conda" init bash
    echo "Miniforge installed. Shell init added to ~/.bashrc."
fi

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------------
# 4. TRex conda environment
# -----------------------------------------------------------------------------
echo ""
echo "--- 4/7  TRex conda environment ('${CONDA_ENV_NAME}') ---"
if conda env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
    echo "Environment '${CONDA_ENV_NAME}' already exists — skipping."
else
    echo "Creating environment and installing TRex (this may take a few minutes)..."
    conda create -n "${CONDA_ENV_NAME}" -c trexing trex -y
    echo "TRex installed."
fi

# -----------------------------------------------------------------------------
# 5. uv
# -----------------------------------------------------------------------------
echo ""
echo "--- 5/7  uv ---"
if command -v uv &>/dev/null; then
    echo "Already installed ($(uv --version)) — skipping."
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "uv installed."
fi

# -----------------------------------------------------------------------------
# 6. Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "--- 6/7  Python dependencies ---"
cd "${INSTALL_DIR}"
uv sync
echo "Dependencies up to date."

# -----------------------------------------------------------------------------
# 7. pipeline.yaml
# -----------------------------------------------------------------------------
echo ""
echo "--- 7/7  pipeline.yaml ---"
PIPELINE_CONFIG="${PIPELINE_DIR}/pipeline.yaml"
PIPELINE_EXAMPLE="${PIPELINE_DIR}/pipeline.yaml.example"
if [ -f "${PIPELINE_CONFIG}" ]; then
    echo "Already exists — skipping."
else
    cp "${PIPELINE_EXAMPLE}" "${PIPELINE_CONFIG}"
    sed -i "s/^conda_env:.*/conda_env: ${CONDA_ENV_NAME}/" "${PIPELINE_CONFIG}"
    echo "Created ${PIPELINE_CONFIG}"
    echo "  conda_env:     ${CONDA_ENV_NAME}"
    echo "  miniforge_dir: ${MINIFORGE_DIR}"
    echo "  Edit this file if your setup differs."
fi

# =============================================================================
echo ""
echo "============================================================"
if [ "${IS_UPDATE}" = true ]; then
    echo "  Update complete."
else
    echo "  Installation complete."
fi
echo "  Repo: ${INSTALL_DIR}"
echo ""
echo "  To update in future:"
echo "    bash ${INSTALL_DIR}/install.sh"
echo ""
echo "  To start a new project:"
echo "    cd ${INSTALL_DIR}"
echo "    uv run tracker new-project --name <project_name> \\"
echo "                                           --path /path/to/projects"
echo ""
echo "  To open the tracker app:"
echo "    uv run tracker --project <project_path>"
echo "============================================================"
