# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MedBridge OpenEnv Server
========================
Serves the MedBridge RL environment using OpenEnv's standard server infrastructure.

This provides:
  - /reset, /step, /state endpoints (OpenEnv protocol)
  - /web interactive UI (OpenEnv built-in Gradio web interface)
  - /health health check
  - /docs OpenAPI documentation
"""

import sys
import os

# Add parent directory to path so we can import medbridge modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.web_interface import create_web_interface_app

from .medbridge_environment import MedbridgeEnvironment

try:
    from ..models import MedbridgeAction, MedbridgeObservation
except ImportError:
    from models import MedbridgeAction, MedbridgeObservation  # type: ignore[import-not-found]


# Create the FastAPI app with OpenEnv web interface.
# This registers /reset, /step, /state, /health AND mounts
# the Gradio-based OpenEnv UI at /web (with / redirecting to /web/).
# Pass the class (factory callable), not an instance.
app = create_web_interface_app(
    env=MedbridgeEnvironment,
    action_cls=MedbridgeAction,
    observation_cls=MedbridgeObservation,
    env_name="medbridge",
)


def main():
    """Entry point for running the MedBridge environment server."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="MedBridge OpenEnv Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
