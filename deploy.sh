#!/usr/bin/env bash
# Restart the WebSocket service after editing the code, then show the log.
set -e
 
SERVICE="pbo-server"
 
echo "==> restarting $SERVICE"
sudo systemctl restart "$SERVICE"
sleep 2
 
if ! systemctl is-active --quiet "$SERVICE"; then
  echo "!! not running — last 30 log lines:"
  journalctl -u "$SERVICE" -n 30 --no-pager
  exit 1
fi
 
echo "==> up. tailing log (Ctrl-C detaches, service keeps running)"
journalctl -u "$SERVICE" -f -n 20
 
