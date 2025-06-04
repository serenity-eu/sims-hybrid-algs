#!/usr/bin/env bash

# Build an app
DOCKER_BUILDKIT=1 docker build -t pls_builder .

# Create temporary container
id=$(docker create pls_builder)

# Copy app from container to host
docker cp $id:/app/target/release/pls pls

# Remove temporary container
docker rm -v $id