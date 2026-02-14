#!/bin/bash
# Force rebuild without Docker cache to ensure latest code is used

cd /opt/tsn-server
sudo docker compose build --no-cache tsn_server
sudo docker compose up -d
