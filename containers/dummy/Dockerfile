FROM python:3.13-alpine

# curl so a service healthcheck can use the natural `CMD curl -f .../v2/health/ready` form
RUN apk add --no-cache curl

COPY server.py /server.py

EXPOSE 8000
# python receives the platform's `--load-model=<name>` args as argv and serves exactly those models
ENTRYPOINT ["python", "-u", "/server.py"]
