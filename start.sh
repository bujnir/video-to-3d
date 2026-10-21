#!/usr/bin/env sh
# macOS / Linux: build (first time only) and start the app, then open the browser.
cd "$(dirname "$0")"
docker compose up --build -d || { echo "Docker is not running. Start Docker and try again."; exit 1; }
sleep 4
(command -v open >/dev/null && open http://localhost:8080) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8080) || true
echo "Video to 3D is running at http://localhost:8080  (stop it with: docker compose down)"
