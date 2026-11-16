#!/usr/bin/bash


docker compose --env-file .env -f ./postgres/docker-compose.yaml down
docker compose --env-file .env -f ./rabbitmq/docker-compose.yaml down
docker compose --env-file .env -f ./minio/docker-compose.yaml down

