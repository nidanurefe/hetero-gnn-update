#!/bin/bash
# =============================================================================
# remote_train.sh - Starts GNN training on remote server via SSH
# =============================================================================
# This script can be called from GitHub Actions or locally.
# Runs training on remote server, exports embeddings, and returns results.
# =============================================================================

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --host          SSH host address (required)
    --user          SSH username (required)
    --path          Remote project path (required)
    --repo          GitHub repository URL (required)
    --branch        Git branch (default: main)
    --model         Model type: gat | sage (default: gat)
    --epochs        Number of epochs (default: 30)
    --batch-size    Batch size (default: 4096)
    --lr            Learning rate (default: 0.002)
    --hidden        Hidden dimension (default: 64)
    --neg-ratio     Negative sample ratio (default: 5.0)
    --exp-name      Experiment name (required)
    --output-dir    Local output directory (default: ./results)
    --help          Show this message

Example:
    $0 --host gpu-server.example.com --user nida --path /home/nida/gnn-project \\
       --repo https://github.com/username/repo.git --model gat --epochs 50 --exp-name experiment_001

EOF
    exit 1
}

# Default values
MODEL="gat"
EPOCHS=30
BATCH_SIZE=4096
LR="0.002"
HIDDEN=64
NEG_RATIO="5.0"
OUTPUT_DIR="./results"
BRANCH="main"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --user) USER="$2"; shift 2 ;;
        --path) REMOTE_PATH="$2"; shift 2 ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --hidden) HIDDEN="$2"; shift 2 ;;
        --neg-ratio) NEG_RATIO="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --help) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Check required parameters
if [[ -z "$HOST" ]] || [[ -z "$USER" ]] || [[ -z "$REMOTE_PATH" ]] || [[ -z "$EXP_NAME" ]] || [[ -z "$REPO_URL" ]]; then
    log_error "Missing required parameters!"
    usage
fi

# SSH connection string
SSH_TARGET="${USER}@${HOST}"
RUN_ID=$(date +%Y%m%d_%H%M%S)
ARTIFACT_ID="${EXP_NAME}-${RUN_ID}"

log_info "========================================"
log_info "  GNN Training Pipeline"
log_info "========================================"
log_info "Host: ${HOST}"
log_info "Path: ${REMOTE_PATH}"
log_info "Repository: ${REPO_URL}"
log_info "Branch: ${BRANCH}"
log_info "Model: ${MODEL}"
log_info "Epochs: ${EPOCHS}"
log_info "Batch Size: ${BATCH_SIZE}"
log_info "Learning Rate: ${LR}"
log_info "Hidden Dim: ${HIDDEN}"
log_info "Experiment: ${EXP_NAME}"
log_info "Artifact ID: ${ARTIFACT_ID}"
log_info "========================================"

# Step 1: Test SSH connection
log_info "Testing SSH connection..."
if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 ${SSH_TARGET} exit; then
    log_error "SSH connection failed! Check your credentials."
    exit 1
fi
log_success "SSH connection successful"

# Step 2: Clone or update repository on remote
log_info "Setting up repository on remote server..."
ssh ${SSH_TARGET} << ENDSSH

echo "🔄 Setting up repository..."
echo "   Repository: ${REPO_URL}"
echo "   Branch: ${BRANCH}"

if [ -d "${REMOTE_PATH}/.git" ]; then
    echo "📂 Repository exists, pulling latest changes..."
    cd "${REMOTE_PATH}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    echo "📥 Cloning repository..."
    mkdir -p "\$(dirname ${REMOTE_PATH})"
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REMOTE_PATH}"
fi

echo "✅ Repository ready!"
cd "${REMOTE_PATH}"
git log -1 --oneline
ENDSSH
log_success "Repository updated"

# Step 3: Create config file
log_info "Creating training configuration on remote server..."
ssh ${SSH_TARGET} << ENDSSH
cd ${REMOTE_PATH}

cat > config/config_pipeline.yaml << 'EOF'
# Auto-generated pipeline config
# Run ID: ${RUN_ID}
# Experiment: ${EXP_NAME}

training:
  graph_path: data/processed/hetero_graph.pt
  save_path: data/processed/models
  epochs: ${EPOCHS}
  batch_size: ${BATCH_SIZE}
  lr: ${LR}
  weight_decay: 0.0001
  num_neighbors: [10,5]
  neg_ratio: ${NEG_RATIO}

model:
  name: ${MODEL}
  hidden: ${HIDDEN}
  heads: 4

loss:
  use_books_loss: 1
  lambda_book: 0.3

export:
  out_dir: data/processed/embeddings
  outfile: "${EXP_NAME}_embeddings.pt"
  device: cuda
EOF

echo "Config created successfully"
ENDSSH

log_success "Configuration created"

# Step 4: Start GNN training
log_info "Starting GNN training..."
ssh ${SSH_TARGET} << ENDSSH
cd ${REMOTE_PATH}
source ~/.bashrc

echo "========================================"
echo "  Starting ${MODEL} training"
echo "========================================"

# Run training script
python src/train_${MODEL}_neighbor.py \
    --config config/config_pipeline.yaml \
    2>&1 | tee logs/training_${RUN_ID}.log

# Rename model checkpoint
MODEL_NAME="${EXP_NAME}_${MODEL}"
if [ -f data/processed/models/best_model.pt ]; then
    cp data/processed/models/best_model.pt data/processed/models/\${MODEL_NAME}.pt
    echo "Model saved as: \${MODEL_NAME}.pt"
fi

echo "Training completed!"
ENDSSH

log_success "Training completed"

# Step 5: Export embeddings
log_info "Exporting embeddings..."
ssh ${SSH_TARGET} << ENDSSH
cd ${REMOTE_PATH}
source ~/.bashrc

cat > config/config_export_pipeline.yaml << 'EOF'
model:
  name: ${MODEL}
  ckpt_path: data/processed/models/${EXP_NAME}_${MODEL}.pt
  hidden: ${HIDDEN}
  heads: 4

export:
  out_dir: data/processed/embeddings
  outfile: "${EXP_NAME}_embeddings.pt"
  device: cuda
  use_checkpoint_sizes: false
EOF

python src/export_embeddings.py --config config/config_export_pipeline.yaml

echo "Embeddings exported!"
ENDSSH

log_success "Embeddings exported"

# Step 6: Download files
log_info "Downloading artifacts..."
mkdir -p ${OUTPUT_DIR}

scp ${SSH_TARGET}:${REMOTE_PATH}/data/processed/embeddings/${EXP_NAME}_embeddings.pt ${OUTPUT_DIR}/
scp ${SSH_TARGET}:${REMOTE_PATH}/data/processed/models/${EXP_NAME}_${MODEL}.pt ${OUTPUT_DIR}/
scp ${SSH_TARGET}:${REMOTE_PATH}/logs/training_${RUN_ID}.log ${OUTPUT_DIR}/

# Parametreleri JSON olarak kaydet
cat > ${OUTPUT_DIR}/params.json << EOF
{
    "experiment_name": "${EXP_NAME}",
    "model_type": "${MODEL}",
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "learning_rate": ${LR},
    "hidden_dim": ${HIDDEN},
    "neg_ratio": ${NEG_RATIO},
    "run_id": "${RUN_ID}",
    "artifact_id": "${ARTIFACT_ID}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

log_success "Artifacts downloaded to: ${OUTPUT_DIR}/"

# Step 7: Summary
log_info "========================================"
log_info "  Pipeline Completed Successfully!"
log_info "========================================"
log_info "Artifact ID: ${ARTIFACT_ID}"
log_info "Files:"
log_info "  - ${OUTPUT_DIR}/${EXP_NAME}_embeddings.pt"
log_info "  - ${OUTPUT_DIR}/${EXP_NAME}_${MODEL}.pt"
log_info "  - ${OUTPUT_DIR}/training_${RUN_ID}.log"
log_info "  - ${OUTPUT_DIR}/params.json"
log_info "========================================"

echo "${ARTIFACT_ID}" > ${OUTPUT_DIR}/artifact_id.txt
