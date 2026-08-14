from celery import Celery

from triton_serve.config.celery import Config

BUILDER_QUEUE = "builder"

app = Celery("serve-sentinel")
app.config_from_object(Config)
