#!/bin/bash
# Diagnostic script to check volume mounts and file visibility

echo "=== CHECKING INGESTION DIRECTORY ==="
echo "Host incoming directory:"
ls -lah /home/wintergrasped/tsn_incoming/ | head -20

echo -e "\n=== CHECKING DOCKER VOLUME ==="
docker volume inspect incoming_data

echo -e "\n=== CHECKING CONTAINER MOUNT ==="
docker exec tsn_server ls -lah /incoming/ | head -20

echo -e "\n=== CHECKING DOCKER COMPOSE VOLUME CONFIG ==="
docker inspect tsn_server | grep -A 10 "Mounts"

echo -e "\n=== FILE COUNTS ==="
echo "Host: $(ls /home/wintergrasped/tsn_incoming/*.{wav,WAV} 2>/dev/null | wc -l) files"
echo "Container: $(docker exec tsn_server sh -c 'ls /incoming/*.{wav,WAV} 2>/dev/null | wc -l') files"
