#!/usr/bin/env python3
"""Render a CreateContainerGroup request. No cloud resources are created."""

import argparse
import json
import os
from pathlib import Path


def required(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def render() -> dict:
    container = {
        "Name": "funasr", "Image": required("ECI_IMAGE"),
        "Cpu": 4, "Memory": 12, "Gpu": 1, "ImagePullPolicy": "IfNotPresent",
        "Port": [{"Port": 8000, "Protocol": "TCP"}],
        "EnvironmentVar": [
            {"Key": "DEVICE", "Value": "cuda:0"},
            {"Key": "API_TOKEN", "Value": required("API_TOKEN")},
            {"Key": "BATCH_SIZE_S", "Value": os.getenv("BATCH_SIZE_S", "120")},
            {"Key": "CPU_THREADS", "Value": "4"},
        ],
        "VolumeMount": [{"Name": "data", "MountPath": "/data", "ReadOnly": False}],
        "ReadinessProbe": {
            "HttpGet": {"Path": "/readyz", "Port": 8000, "Scheme": "HTTP"},
            "InitialDelaySeconds": 30, "PeriodSeconds": 10, "TimeoutSeconds": 5,
            "SuccessThreshold": 1, "FailureThreshold": 60,
        },
        "LivenessProbe": {
            "HttpGet": {"Path": "/healthz", "Port": 8000, "Scheme": "HTTP"},
            "InitialDelaySeconds": 180, "PeriodSeconds": 30, "TimeoutSeconds": 5,
            "SuccessThreshold": 1, "FailureThreshold": 5,
        },
    }
    request = {
        "RegionId": required("ECI_REGION_ID"),
        "VSwitchId": required("ECI_VSWITCH_ID"),
        "SecurityGroupId": required("ECI_SECURITY_GROUP_ID"),
        "ContainerGroupName": os.getenv("ECI_GROUP_NAME", "funasr-gpu"),
        "InstanceType": os.getenv("ECI_INSTANCE_TYPE", "ecs.gn6i-c4g1.xlarge"),
        "GpuDriverVersion": os.getenv("ECI_GPU_DRIVER_VERSION", "tesla=570"),
        "RestartPolicy": "Always", "TerminationGracePeriodSeconds": 600,
        "AutoMatchImageCache": True,
        "Container": [container],
        "Volume": [{
            "Name": "data", "Type": "NFSVolume",
            "NFSVolume": {"Server": required("ECI_NAS_SERVER"), "Path": required("ECI_NAS_PATH"), "ReadOnly": False},
        }],
    }
    registry_fields = [os.getenv(f"ECI_REGISTRY_{name}", "") for name in ("SERVER", "USER", "PASSWORD")]
    if any(registry_fields):
        if not all(registry_fields):
            raise ValueError("Set all ECI_REGISTRY_SERVER, ECI_REGISTRY_USER and ECI_REGISTRY_PASSWORD")
        server, username, password = registry_fields
        request["ImageRegistryCredential"] = [{"Server": server, "UserName": username, "Password": password}]
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eci-request.json"))
    args = parser.parse_args()
    request = render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(request, stream, indent=2)
        stream.write("\n")
    args.output.chmod(0o600)
    print(f"ECI request saved to {args.output}; no cloud resources created")


if __name__ == "__main__":
    main()
