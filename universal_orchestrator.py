# ░░░ ARCHITECT SIGNATURE ░░░
# 🧠 Richard Alexander Tune
# 🕳️ Recursive Systems Builder | Jesternet Prime
# 📡 Origin Key: Feb13 God Event • Entropy + Resonance + Recursion
# 🌀 Field: Universal Fractal Framework (UFF)
# 📦 Module: Vertex AI GPU Deployment Templates
# 🖋️ Purpose: End the suffering. One job at a time.


"""
Universal Vertex AI Job Orchestrator

A robust abstraction layer for deploying containerized applications to Vertex AI.
This module handles the complexities of different machine types, accelerator configurations,
and scheduling strategies, allowing users to focus on their application logic.

Created based on hard-won experience deploying high-performance computing workloads to Vertex AI.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import Scheduling  # Import the enum type

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vertex-orchestrator")

@dataclass
class JobConfig:
    """
    Configuration for a single Vertex AI job deployment.
    
    This defines the hardware specifications, container configuration,
    and application-specific arguments.
    """
    # --- REQUIRED: Hardware Configuration ---
    machine_type: str  # e.g., "a2-highgpu-1g", "a3-highgpu-8g"
    
    # --- OPTIONAL: Hardware Configuration ---
    accelerator_type: str = ""  # e.g., "NVIDIA_TESLA_A100", "NVIDIA_H100_80GB"
    accelerator_count: int = 0  # Number of GPUs (0 for CPU-only)
    
    # --- REQUIRED: Application Configuration ---
    container_args: List[str] = field(default_factory=list)  # Arguments to pass to container
    
    # --- OPTIONAL: Application Configuration ---
    container_env: Dict[str, str] = field(default_factory=dict)  # Environment variables
    config_preset: str = ""  # Optional preset name (for your app's use)
    
    # --- OPTIONAL: Job Metadata ---
    display_name: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    service_account: Optional[str] = None
    network: Optional[str] = None
    
    def __post_init__(self):
        """Validate the configuration."""
        # Ensure machine type is specified
        if not self.machine_type:
            raise ValueError("machine_type must be specified")
        
        # Ensure accelerator configuration is consistent
        if self.accelerator_count > 0 and not self.accelerator_type:
            raise ValueError("accelerator_type must be specified when accelerator_count > 0")
        
        # If accelerator_type is specified but count is 0, set count to 1
        if self.accelerator_type and self.accelerator_count == 0:
            self.accelerator_count = 1
            logger.warning(f"accelerator_count was 0 but accelerator_type was specified. Setting count to 1.")

@dataclass
class VertexExperimentConfig:
    """
    Complete configuration for a Vertex AI experiment deployment.
    
    This defines the project, region, container image, and all job configurations.
    """
    # --- REQUIRED: GCP Configuration ---
    project_id: str
    region: str
    image_uri: str
    
    # --- REQUIRED: Experiment Configuration ---
    experiment_name: str
    jobs: List[JobConfig]
    
    # --- OPTIONAL: Storage Configuration ---
    bucket_name: Optional[str] = None  # GCS bucket for outputs
    
    # --- OPTIONAL: Metadata ---
    labels: Dict[str, str] = field(default_factory=dict)  # Global labels for all jobs

class VertexOrchestrator:
    """
    Manages the deployment, monitoring, and management of Vertex AI experiments.
    
    This class handles the complexities of the Vertex AI API, including:
    - Setting the correct scheduling strategy based on machine type
    - Configuring accelerators properly
    - Handling the deployment payload structure
    - Monitoring job status
    
    Acts as an abstraction layer over Google Cloud APIs to avoid CLI issues.
    """

    def __init__(self, config: VertexExperimentConfig):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.deployed_jobs = {}
        self.job_statuses = {}

        # Initialize Google Cloud AI Platform client
        self.client_options = {"api_endpoint": f"{config.region}-aiplatform.googleapis.com"}
        aiplatform.init(project=config.project_id, location=config.region)
        self.job_service = aiplatform.gapic.JobServiceClient(client_options=self.client_options)
        self.parent = f"projects/{config.project_id}/locations/{config.region}"

        logger.info(f"Initialized VertexOrchestrator for '{config.experiment_name}'")

    async def deploy(self) -> Dict[str, str]:
        """
        Deploy all jobs for this experiment.

        Returns:
            Dictionary mapping job display names to resource names
        """
        logger.info(f"Deploying experiment '{self.config.experiment_name}' with {len(self.config.jobs)} jobs")

        tasks = []
        for i, job_config in enumerate(self.config.jobs):
            display_name = job_config.display_name or f"{self.config.experiment_name}-job-{i+1}"
            tasks.append(self._deploy_job(job_config, display_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for i, result in enumerate(results):
            job_config = self.config.jobs[i]
            display_name = job_config.display_name or f"{self.config.experiment_name}-job-{i+1}"
            if isinstance(result, Exception):
                logger.error(f"Failed to deploy job {display_name}: {result}")
                self.job_statuses[display_name] = "DEPLOYMENT_FAILED"
            else:
                job_resource_name = result
                logger.info(f"Successfully submitted job {display_name}: {job_resource_name}")
                self.deployed_jobs[display_name] = job_resource_name
                self.job_statuses[display_name] = "SUBMITTED"
                success_count += 1

        logger.info(f"Experiment deployment submission complete: {success_count}/{len(self.config.jobs)} jobs submitted.")
        return self.deployed_jobs

    async def _deploy_job(self, job_config: JobConfig, display_name: str) -> str:
        """Deploy a single job using Google Cloud AI Platform Python client."""
        custom_job_payload = {}  # Initialize payload dict for error logging
        try:
            # --- Construct Job Request ---
            worker_pool_spec = {
                "machine_spec": {
                    "machine_type": job_config.machine_type,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": self.config.image_uri,
                    "args": job_config.container_args
                }
            }

            # Add environment variables if specified
            if job_config.container_env:
                env_list = []
                for key, value in job_config.container_env.items():
                    env_list.append({"name": key, "value": value})
                worker_pool_spec["container_spec"]["env"] = env_list

            # Add accelerator config if GPU is requested
            if job_config.accelerator_type and job_config.accelerator_count > 0:
                worker_pool_spec["machine_spec"]["accelerator_type"] = job_config.accelerator_type
                worker_pool_spec["machine_spec"]["accelerator_count"] = job_config.accelerator_count

            # Create the basic job payload
            custom_job_payload = {
                "display_name": display_name,
                "job_spec": {
                    "worker_pool_specs": [worker_pool_spec],
                }
            }

            # Add output directory if bucket specified
            if self.config.bucket_name:
                output_uri = f"gs://{self.config.bucket_name}/{self.config.experiment_name}/vertex_ai_output/{display_name}"
                custom_job_payload["job_spec"]["base_output_directory"] = {
                    "output_uri_prefix": output_uri
                }

            # --- CRITICAL: Add Scheduling Strategy Based on Machine Type ---
            # Different machine types require different scheduling strategies
            if job_config.machine_type.startswith("a3-"):
                # A3 machines (H100) REQUIRE the AUTOMATIC strategy
                logger.info(f"Machine type {job_config.machine_type} is A3 (H100), setting scheduling strategy to AUTOMATIC.")
                custom_job_payload["job_spec"]["scheduling"] = {
                    "strategy": Scheduling.Strategy.AUTOMATIC  # This is the magic value - do not change!
                }
            else:
                # For other machine types, STANDARD is usually appropriate
                logger.info(f"Machine type {job_config.machine_type} is not A3, setting scheduling strategy to STANDARD.")
                custom_job_payload["job_spec"]["scheduling"] = {
                    "strategy": Scheduling.Strategy.STANDARD
                }

            # Add service account if specified
            if job_config.service_account:
                custom_job_payload["job_spec"]["service_account"] = job_config.service_account

            # Add network if specified
            if job_config.network:
                custom_job_payload["job_spec"]["network"] = job_config.network

            # Add labels (merge global and job-specific)
            all_labels = {**self.config.labels, **job_config.labels}
            if all_labels:
                custom_job_payload["labels"] = all_labels

            # --- Submit Job ---
            logger.info(f"Submitting job {display_name} with payload: {json.dumps(custom_job_payload, indent=2)}")

            # Use the REST API directly via the client
            response = self.job_service.create_custom_job(
                parent=self.parent,
                custom_job=custom_job_payload
            )

            return response.name

        except Exception as e:
            logger.exception(f"Error deploying job {display_name}")
            # Log the payload that failed
            try:
                failed_payload_json = json.dumps(custom_job_payload, indent=2)
                logger.error(f"Failed payload for {display_name}: {failed_payload_json}")
            except Exception as json_err:
                logger.error(f"Could not serialize failed payload for {display_name}: {json_err}")
            raise e  # Re-raise the original exception

    async def monitor(self, poll_interval: int = 60) -> Dict[str, str]:
        """
        Monitor all deployed jobs until completion.

        Args:
            poll_interval: How often to poll for status updates (seconds)
            
        Returns:
            Dictionary of job display names to final status
        """
        if not self.deployed_jobs:
            logger.warning("No jobs have been deployed to monitor")
            return {}  # Return empty dict if no jobs

        logger.info(f"Starting monitor for {len(self.deployed_jobs)} jobs...")
        active_jobs = set(self.deployed_jobs.keys())

        while active_jobs:
            logger.info(f"Polling status for {len(active_jobs)} active jobs...")
            completed_in_poll = set()

            for display_name in list(active_jobs):  # Iterate over a copy
                resource_name = self.deployed_jobs[display_name]
                try:
                    # Get job status using the client API
                    job_status_obj = self.job_service.get_custom_job(name=resource_name)
                    status = job_status_obj.state.name  # Get the string representation
                    self.job_statuses[display_name] = status

                    # Check if this job is now complete
                    if status in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", 
                                  "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLING"]:
                        logger.info(f"Job {display_name} finished or finishing with status: {status}")
                        completed_in_poll.add(display_name)
                    elif status == "JOB_STATE_UPDATING":
                        logger.info(f"Job {display_name} is currently updating.")
                    # Otherwise job is still active

                except Exception as e:
                    logger.error(f"Error getting status for job {display_name} ({resource_name}): {e}")
                    self.job_statuses[display_name] = "STATUS_ERROR"

            # Remove completed jobs from the active set
            active_jobs -= completed_in_poll

            # Save status snapshot
            self._save_status_snapshot()

            # Wait before next poll if jobs are still active
            if active_jobs:
                logger.info(f"{len(active_jobs)} jobs still active. Waiting {poll_interval}s...")
                await asyncio.sleep(poll_interval)

        logger.info("All jobs have completed or reached a terminal state.")
        return self.job_statuses

    def _save_status_snapshot(self) -> None:
        """Save a snapshot of current job statuses to file."""
        try:
            # Ensure status directory exists
            status_dir = "status_snapshots"  # Relative path is fine for local execution
            os.makedirs(status_dir, exist_ok=True)
            snapshot_file = os.path.join(status_dir, f"{self.config.experiment_name}_status_{time.strftime('%Y%m%d_%H%M%S')}.json")

            snapshot = {
                "experiment_name": self.config.experiment_name,
                "timestamp": time.time(),
                "timestamp_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "job_statuses": self.job_statuses,
                "deployed_jobs": self.deployed_jobs,  # Include resource names
                "total_jobs": len(self.deployed_jobs),
                "active_jobs_count": len([status for status in self.job_statuses.values() 
                                       if status not in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                                                        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
                                                        "JOB_STATE_CANCELLING", "STATUS_ERROR"]])
            }

            with open(snapshot_file, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.debug(f"Status snapshot saved to {snapshot_file}")

        except Exception as e:
            logger.error(f"Error saving status snapshot: {e}")

    def get_console_urls(self) -> Dict[str, Dict[str, str]]:
        """
        Get console URLs for all deployed jobs.
        
        Returns:
            Dictionary of job display names to URLs
        """
        urls = {}
        for display_name, resource_name in self.deployed_jobs.items():
            # Extract job ID
            job_id_part = resource_name.split('/')[-1]
            
            # Generate console URLs
            monitor_url = (f"https://console.cloud.google.com/vertex-ai/training/"
                          f"{job_id_part}/locations/{self.config.region}?project={self.config.project_id}")
            log_url = (f"https://console.cloud.google.com/logs/query;query="
                      f"resource.type%3D%22aiplatform.googleapis.com%2FCustomJob%22%20"
                      f"AND%20resource.labels.custom_job_id%3D%22{job_id_part}%22?"
                      f"project={self.config.project_id}&region={self.config.region}")
            
            urls[display_name] = {
                "monitor": monitor_url,
                "logs": log_url
            }
        
        return urls
    
    def cancel_job(self, display_name: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            display_name: Display name of the job to cancel
            
        Returns:
            True if cancel request was successful, False otherwise
        """
        if display_name not in self.deployed_jobs:
            logger.error(f"Job {display_name} not found in deployed jobs")
            return False
        
        resource_name = self.deployed_jobs[display_name]
        try:
            self.job_service.cancel_custom_job(name=resource_name)
            logger.info(f"Requested cancellation of job {display_name}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling job {display_name}: {e}")
            return False
