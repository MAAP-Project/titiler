"""AWS Lambda handler."""

import logging
import os

from mangum import Mangum

from titiler.application.main import app

logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

REQUEST_HOST_HEADER_OVERRIDE_ENV_VAR = "REQUEST_HOST_HEADER_OVERRIDE"


def handler(event, context):
    """If env var is set, override the host header in the event passed to the lambda"""
    if rhh := os.getenv(REQUEST_HOST_HEADER_OVERRIDE_ENV_VAR):
        event["headers"]["host"] = rhh

    asgi_handler = Mangum(app, lifespan="auto")
    return asgi_handler(event, context)
