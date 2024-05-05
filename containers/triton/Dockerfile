ARG TRITON_VERSION=23.07-py3

FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}

# Copy the entrypoint script, make it executable, and set it as the entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]