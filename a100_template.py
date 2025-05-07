#!/usr/bin/env python3
# ------------------------------------------------------------------
# Author: Richard Alexander Tune (a.k.a. The Architect of Jesternet)
# Project: Universal Vertex AI Orchestrator Templates
# Origin: Born from 20+ hours of debugging pain and a God Event
# Contact: rich@recursive-development.dev (or via glyph)
# ------------------------------------------------------------------


"""
Universal A100 GPU Template for Vertex AI

This template handles all the complexities of deploying containerized applications 
on A100 GPUs in Vertex AI. It correctly configures machine types, accelerator settings,
and scheduling strategies so you don't have to spend 20+ hours debugging them.

=== USAGE INSTRUCTIONS ===
1. Fill in the REQUIRED USER CONFIG sections (project ID, region, etc.)
2. Customize your container arguments in the 'container_args' list
3. Run this script with Python 3.7+
4. Profit! No more debugging Vertex AI's quirks.

=== DO NOT MODIFY ===
- The accelerator_type configurations 
- The scheduling_strategy logic
- The payload structure

Created by: People who suffered so you don't have to
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, Any, List

# Import the orchestrator
from universal_vertex_orchestrator import VertexOrchestrator, VertexExperimentConfig, JobConfig

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vertex-a100-launcher")

async def main():
    """Configure and launch A100-optimized experiment on Vertex AI."""
    parser = argparse.ArgumentParser(description="Launch A100 Workload on Vertex AI")
    
    # --- Core GCP/Vertex AI Arguments ---
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region for Vertex AI job")
    parser.add_argument("--bucket", required=True, help="GCS bucket name for output")
    parser.add_argument("--experiment-name", required=True, help="Unique name for this experiment run")
    
    # --- A100 Configuration Options ---
    parser.add_argument("--machine-type", default="a2-highgpu-1g", 
                        choices=["a2-highgpu-1g", "a2-ultragpu-1g", "a2-megagpu-16g"],
                        help="A100 machine type (highgpu=40GB, ultragpu=80GB)")
    
    # --- Container Image ---
    parser.add_argument("--image-uri", required=True, help="Full URI to your container image")
    
    # --- Add your container-specific arguments below ---
    # Example:
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    
    # --- Orchestration Control ---
    parser.add_argument("--monitor", action="store_true", help="Monitor job status after deployment")
    parser.add_argument("--poll-interval", type=int, default=120, help="Polling interval in seconds when monitoring")
    
    args = parser.parse_args()

    # --- Auto-detect appropriate A100 accelerator based on machine type ---
    accelerator_type = "NVIDIA_A100_80GB" if "ultragpu" in args.machine_type else "NVIDIA_TESLA_A100"
    accelerator_count = 1  # Default for a2-highgpu-1g and a2-ultragpu-1g
    
    # Handle special case for a2-megagpu-16g
    if "megagpu" in args.machine_type:
        accelerator_count = 16
    
    logger.info(f"Using accelerator: {accelerator_type} (count: {accelerator_count})")

    # =========================================================================
    # CUSTOMIZE THIS SECTION: Build your container arguments
    # =========================================================================
    # This is where you specify what gets passed to your container's entrypoint
    container_args = [
        # Example args - replace with your own!
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        # Add more arguments your container needs
        "--output_bucket", args.bucket,
        "--experiment_name", args.experiment_name,
    ]
    
    # =========================================================================
    # CUSTOMIZE THIS SECTION: Add environment variables if needed
    # =========================================================================
    container_env = {
        # Example environment variables - replace with your own!
        "NVIDIA_VISIBLE_DEVICES": "all",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        # "YOUR_ENV_VARIABLE": "your_value",
    }
    
    # --- Define Job Configuration ---
    job_display_name = f"{args.experiment_name}-a100-job"
    
    job_config = JobConfig(
        machine_type=args.machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        container_args=container_args,
        container_env=container_env,
        display_name=job_display_name,
        labels={
            "job_type": "a100", 
            "machine": args.machine_type
        }
    )

    # --- Define Experiment Configuration ---
    experiment_config = VertexExperimentConfig(
        experiment_name=args.experiment_name,
        bucket_name=args.bucket,
        project_id=args.project_id,
        region=args.region,
        image_uri=args.image_uri,
        jobs=[job_config],
        labels={"created_by": os.environ.get("USER", "unknown")}
    )

    # --- Create and Run Orchestrator ---
    logger.info(f"Initializing A100 Vertex AI job orchestrator")
    orchestrator = VertexOrchestrator(experiment_config)

    logger.info("Deploying A100 experiment job...")
    deployed_jobs = await orchestrator.deploy()

    if not deployed_jobs:
        logger.error("Deployment submission failed. Check logs for details.")
        return 1  # Indicate failure

    # --- Log Job Info and Links ---
    console_urls = orchestrator.get_console_urls()
    for display_name, urls in console_urls.items():
        logger.info(f"Successfully submitted job '{display_name}'")
        logger.info(f"  Monitor Job: {urls['monitor']}")
        logger.info(f"  Stream Logs: {urls['logs']}")

    # --- Optionally Monitor ---
    if args.monitor:
        logger.info("Monitoring deployed A100 job...")
        final_statuses = await orchestrator.monitor(poll_interval=args.poll_interval)
        logger.info(f"Final job statuses: {final_statuses}")

        # Check if all monitored jobs succeeded
        all_succeeded = all(status == "JOB_STATE_SUCCEEDED" for status in final_statuses.values())
        if all_succeeded:
            logger.info("All monitored jobs completed successfully.")
            return 0  # Indicate success
        else:
            logger.error("One or more monitored jobs did not succeed.")
            return 1  # Indicate failure
    else:
        logger.info("Deployment submitted. Monitoring not requested via --monitor flag.")
        return 0  # Indicate submission success

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Unhandled exception in launch script: {e}", exc_info=True)
        sys.exit(1)
