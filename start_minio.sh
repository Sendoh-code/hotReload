#!/bin/bash
MINIO_ROOT_USER=minioadmin \
MINIO_ROOT_PASSWORD=minioadmin \
MINIO_DOMAIN=localhost \
minio server ~/minio-data \
  --console-address :9001
