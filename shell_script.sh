#!/bin/bash
# Universal GPU Deployment Script for Vertex AI
# Based on hard-earned experience of what actually works

set -e  # Exit on error

# =========================================================================
# REQUIRED USER CONFIG - Change these values!
# =========================================================================
PROJECT_ID="your-project-id"        # CHANGE THIS
REGION="us-central1"                # CHANGE THIS: A100: us-central1, H100: us-west1
BUCKET="your-gcs-bucket"            # CHANGE THIS
EXPERIMENT="my-gpu-job-$(date +%Y%m%d_%H%M)"

# Choose GPU configuration - A100 or H100
GPU_TYPE="a100"  # CHANGE THIS: Options: "a100" or "h100"

# =========================================================================
# CONTAINER CONFIG - Change these values!
# =========================================================================
# Your container image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/your-repo/your-image:latest"  # CHANGE THIS

# =========================================================================
# GPU Configuration - DO NOT CHANGE unless you know what you're doing!
# =========================================================================
if [ "$GPU_TYPE" = "a100" ]; then
    # A100 Configuration - Single 80GB
    MACHINE_TYPE="a2-ultragpu-1g"
    ACCELERATOR="NVIDIA_A100_80GB"
    ACCELERATOR_COUNT=1
    STRATEGY="STANDARD"  # Standard is fine for A2 machines
elif [ "$GPU_TYPE" = "h100" ]; then
    # H100 Configuration - 8x H100 80GB
    MACHINE_TYPE="a3-highgpu-8g"
    ACCELERATOR="NVIDIA_H100_80GB"
    ACCELERATOR_COUNT=8
    STRATEGY="AUTOMATIC"  # CRITICAL: A3 machines require AUTOMATIC
else
    echo "Error: Unknown GPU_TYPE. Use 'a100' or 'h100'"
    exit 1
fi

# --- Display Configuration ---
echo "==================================================="
echo "Vertex AI GPU Deployment"
echo "  Project ID:      $PROJECT_ID"
echo "  Region:          $REGION"
echo "  Bucket:          $BUCKET"
echo "  Experiment:      $EXPERIMENT"
echo "  GPU Type:        $GPU_TYPE"
echo "  Machine Type:    $MACHINE_TYPE"
echo "  Accelerator:     $ACCELERATOR (count: $ACCELERATOR_COUNT)"
echo "  Strategy:        $STRATEGY"
echo "  Container:       $IMAGE_URI"
echo "==================================================="
echo ""

# --- GCP Setup ---
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

# Ensure GCS bucket exists
echo "Checking GCS bucket: $BUCKET"
if ! gsutil ls -b gs://$BUCKET &> /dev/null; then
    echo "Creating GCS bucket..."
    gsutil mb -l $REGION gs://$BUCKET
else
    echo "Using existing GCS bucket."
fi

# --- Create gcloud JSON file for job specification ---
CONFIG_FILE="vertex_gpu_job_config.json"

# =========================================================================
# CONTAINER ARGS - Change these values to match your container needs!
# =========================================================================
cat > $CONFIG_FILE << EOL
{
  "displayName": "vertex-gpu-${EXPERIMENT}",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "${MACHINE_TYPE}",
          "acceleratorType": "${ACCELERATOR}",
          "acceleratorCount": ${ACCELERATOR_COUNT}
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "${IMAGE_URI}",
          "args": [
            "--parameter1=value1",
            "--parameter2=value2",
            "--output_bucket=${BUCKET}",
            "--experiment_name=${EXPERIMENT}"
          ],
          "env": [
            {
              "name": "NVIDIA_VISIBLE_DEVICES",
              "value": "all"
            }
          ]
        }
      }
    ],
    "scheduling": {
      "strategy": "${STRATEGY}"
    },
    "baseOutputDirectory": {
      "outputUriPrefix": "gs://${BUCKET}/${EXPERIMENT}/vertex_ai_output"
    }
  }
}
EOL

echo "Created job configuration file: $CONFIG_FILE"

# --- Create Vertex AI Custom Job ---
echo "Creating Vertex AI custom job..."
JOB_ID=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --config=$CONFIG_FILE \
  --format="value(name)")

# Extract job number from full ID
JOB_NUMBER=$(basename $JOB_ID)

echo "Vertex AI custom job created successfully!"
echo "Job Name: $JOB_ID"
echo "Job Number: $JOB_NUMBER"

echo "=================================================="
CONSOLE_URL="https://console.cloud.google.com/vertex-ai/training/custom-jobs/$JOB_NUMBER/detail?project=$PROJECT_ID"
echo "View job status: $CONSOLE_URL"
echo ""
echo "To monitor logs, run:"
echo "gcloud ai custom-jobs stream-logs $JOB_NUMBER --region=$REGION --project=$PROJECT_ID"
echo ""

# --- Stream logs automatically ---
echo "Streaming logs automatically... (Ctrl+C to stop)"
sleep 10  # Give the job time to start
gcloud ai custom-jobs stream-logs $JOB_NUMBER --region=$REGION --project=$PROJECT_ID

echo ""
echo "Deployment complete!"


# ------------------------------------------------------------------
# Author: Richard Alexander Tune (a.k.a. The Architect of Jesternet)
# Project: Universal Vertex AI Orchestrator Templates
# Origin: Born from 20+ hours of debugging pain
# Contact: rich@recursive-development.dev (or via glyph)
# ------------------------------------------------------------------
