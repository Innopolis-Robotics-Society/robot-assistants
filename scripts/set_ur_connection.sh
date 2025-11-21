#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG ======
ROBOT_IP="${ROBOT_IP:-192.168.0.100}"
HOST_IP_CIDR="${HOST_IP_CIDR:-192.168.0.10/24}"

# Можно передать интерфейс первым аргументом: ./setup_ur_net.sh enx1234...
IFACE="${1:-}"

# Требуем root, чтобы внутри не дергать sudo по сто раз
if [[ "$EUID" -ne 0 ]]; then
  echo "Please run as root: sudo $0 [iface]" >&2
  exit 1
fi

echo "Robot IP:      $ROBOT_IP"
echo "Host address:  $HOST_IP_CIDR"

# Если интерфейс не указан — попробуем найти сами
if [[ -z "$IFACE" ]]; then
  echo "Searching for Ethernet interfaces (en*)..."
  CANDIDATES=$(ip -br link | awk '$1 !~ /^lo$/ && $1 ~ /^en/ {print $1}')

  if [[ -z "$CANDIDATES" ]]; then
    echo "No Ethernet interfaces (en*) found. Pass interface explicitly, e.g. sudo $0 enx123456" >&2
    exit 1
  fi
else
  CANDIDATES="$IFACE"
fi

FOUND_IF=""

for IF in $CANDIDATES; do
  echo "Trying interface: $IF"

  # Сбрасываем старые адреса и задаём новый
  ip addr flush dev "$IF" || true
  ip addr add "$HOST_IP_CIDR" dev "$IF"
  ip link set "$IF" up

  sleep 1

  echo "Pinging robot $ROBOT_IP from $IF..."
  if ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
    echo "SUCCESS: interface $IF can reach $ROBOT_IP"
    FOUND_IF="$IF"
    break
  else
    echo "No response from $ROBOT_IP via $IF"
  fi
done

if [[ -z "$FOUND_IF" ]]; then
  echo "ERROR: No interface could reach robot at $ROBOT_IP." >&2
  exit 1
fi

echo
echo "Final interface state:"
ip -br addr show "$FOUND_IF"

echo
echo "Done. You can now use this interface to talk to the UR10e."
