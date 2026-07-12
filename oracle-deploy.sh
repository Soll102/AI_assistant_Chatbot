#!/bin/bash
set -e

REPO_URL="https://github.com/Soll102/AI_assistant_Chatbot.git"
APP_DIR="$HOME/rag_chatbot"
BACKEND_PORT=8000

echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y docker.io nginx

echo "=== Cloning repo ==="
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "=== Building Docker image ==="
sudo docker build -t rag-chatbot-backend .

echo "=== Stopping existing container ==="
sudo docker rm -f rag-chatbot 2>/dev/null || true

echo "=== Running backend ==="
mkdir -p "$APP_DIR/hf_cache"
sudo docker run -d \
    --name rag-chatbot \
    --restart unless-stopped \
    -p 127.0.0.1:$BACKEND_PORT:$BACKEND_PORT \
    -v "$APP_DIR/backend/storage:/app/storage" \
    -v "$APP_DIR/hf_cache:/root/.cache/huggingface" \
    -e GEMINI_API_KEY="your_api_key_here" \
    -e GEMINI_MODEL="gemini-2.5-flash-lite" \
    -e BACKEND_CORS_ORIGINS="https://soll102.github.io" \
    rag-chatbot-backend

echo "=== Configuring nginx reverse proxy ==="
sudo tee /etc/nginx/sites-available/rag-chatbot > /dev/null << 'NGINX'
server {
    listen 80;
    server_name _;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/rag-chatbot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<your-oracle-vm-public-ip>")

echo ""
echo "=========================================="
echo "  Deployment complete!"
echo "=========================================="
echo ""
echo "Backend URL: http://$PUBLIC_IP:$BACKEND_PORT"
echo "Health check: http://$PUBLIC_IP:$BACKEND_PORT/health"
echo ""
echo "IMPORTANT:"
echo "  1. Update GEMINI_API_KEY:"
echo "     sudo docker stop rag-chatbot"
echo "     sudo docker rm rag-chatbot"
echo "     # Then re-run the docker run command above with your actual key"
echo ""
echo "  2. In your Oracle Cloud dashboard, add an ingress rule for port 80:"
echo "     Networking -> Virtual Cloud Networks -> (your VCN) -> Security Lists"
echo "     -> Ingress Rules -> Add: Source 0.0.0.0/0, Destination Port 80"
echo ""
echo "  3. Set VITE_API_BASE in GitHub repo:"
echo "     Settings -> Secrets and variables -> Actions -> Variables"
echo "     -> Add VITE_API_BASE = http://$PUBLIC_IP"
echo ""
