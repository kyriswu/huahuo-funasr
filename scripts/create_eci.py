#!/usr/bin/env python3
"""Apply a reviewed ECI request. Running this script creates a billable GPU instance."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--apply", action="store_true", help="Actually create the ECI container group")
    args = parser.parse_args()
    data = json.loads(args.request.read_text())
    if not args.apply:
        print("Request parsed. Add --apply to create the GPU container group.")
        return
    from alibabacloud_credentials.client import Client as CredentialsClient
    from alibabacloud_eci20180808.client import Client
    from alibabacloud_eci20180808.models import CreateContainerGroupRequest
    from alibabacloud_tea_openapi.models import Config

    config = Config(credential=CredentialsClient(), region_id=data["RegionId"])
    config.endpoint = f"eci.{data['RegionId']}.aliyuncs.com"
    request = CreateContainerGroupRequest().from_map(data)
    response = Client(config).create_container_group(request)
    print(json.dumps(response.body.to_map(), indent=2))


if __name__ == "__main__":
    main()
