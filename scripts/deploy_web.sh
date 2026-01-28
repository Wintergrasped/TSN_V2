#!/bin/bash
# Quick deployment script - run on Linux server after pushing changes

echo "🚀 Deploying TSN V2 updates..."

# Pull latest code
echo "📥 Pulling latest code from GitHub..."
git pull origin main

# Rebuild and restart web container
echo "🔨 Rebuilding web container..."
docker-compose build tsn_web

echo "🔄 Restarting web container..."
docker-compose up -d tsn_web

# Wait for startup
echo "⏳ Waiting for startup (10 seconds)..."
sleep 10

# Check logs
echo "📋 Recent logs:"
docker-compose logs --tail=30 tsn_web

echo ""
echo "✅ Deployment complete!"
echo "🌐 Check: https://tsn.kk7nqn.net/net-control"
