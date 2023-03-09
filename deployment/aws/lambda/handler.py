"""AWS Lambda handler."""

import logging

from mangum import Mangum

from titiler.application.main import app

logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

# mangum > 0.11.x removes the "log_level" parameter from the Mangum constructor
handler = Mangum(app, lifespan="auto")
