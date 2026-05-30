# G1 DDS Connection Notes

This file records the current wired connection workflow for the Unitree G1 from
WSL/Linux. Use read-only state checks first. Do not run motion until DDS state is
confirmed.

## Known Interfaces

Simulator DDS:

```bash
--interface eth3
--domain-id 1
```

Real G1 wired DDS:

```bash
--interface eth0
--domain-id 0
```

Current real robot host IP:

```text
eth0 = 192.168.123.222/24
```

## After Plugging In The Robot Cable

On Windows, replugging the cable can reset the Ethernet network profile back to
`Public`. Set the robot Ethernet adapter to `Private` and allow local DDS UDP
traffic.

Run in **Administrator PowerShell**. Current adapter index on this machine is
`16`:

```powershell
Get-NetConnectionProfile -InterfaceIndex 16
Set-NetConnectionProfile -InterfaceIndex 16 -NetworkCategory Private

New-NetFirewallRule `
  -DisplayName "Unitree DDS UDP In" `
  -Direction Inbound `
  -Action Allow `
  -Protocol UDP `
  -RemoteAddress 192.168.123.0/24 `
  -Profile Private

New-NetFirewallRule `
  -DisplayName "Unitree DDS UDP Out" `
  -Direction Outbound `
  -Action Allow `
  -Protocol UDP `
  -RemoteAddress 192.168.123.0/24 `
  -Profile Private
```

For a temporary diagnosis only, you can disable the Private firewall while the
robot cable is connected:

```powershell
Set-NetFirewallProfile -Profile Private -Enabled False
```

Turn it back on after the test:

```powershell
Set-NetFirewallProfile -Profile Private -Enabled True
```

Check Linux-side interfaces:

```bash
ip -brief addr
ip route
ip route get 239.255.0.1
```

For real G1, `eth0` should have `192.168.123.222/24`.

If `eth0` lost its IP, restore it:

```bash
sudo ip addr flush dev eth0
sudo ip addr add 192.168.123.222/24 dev eth0
sudo ip link set eth0 up
```

Force DDS multicast discovery to the robot cable:

```bash
sudo ip route del 224.0.0.0/4 2>/dev/null || true
sudo ip route add 224.0.0.0/4 dev eth0
ip route get 239.255.0.1
```

Expected route:

```text
multicast 239.255.0.1 dev eth0 src 192.168.123.222
```

If it says `dev eth3`, DDS discovery is going to the simulator/network
interface instead of the real robot cable.

## Read-Only DDS State Check

Always run this before motion:

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python apps/dump_state.py \
  --interface eth0 \
  --domain-id 0 \
  --timeout 10
```

Success means `rt/lowstate` was received and joint positions printed.

If no sample arrives, inspect DDS traffic:

```bash
sudo tcpdump -ni eth0 'udp and (portrange 7400-7600 or multicast)'
```

You should see UDP traffic from the robot or DDS multicast traffic on `eth0`.

## Real Arm-SDK Trajectory Test

Only after `dump_state.py` works:

```bash
cd ~/workspace/mochi
source .venv/bin/activate

python apps/replay_trajectory.py \
  --interface eth0 \
  --domain-id 0 \
  --trajectory dual_hold_swing_v0 \
  --arm-sdk \
  --max-step-rad 0.003
```

SPACE workflow:

```text
SPACE 1: ramp from current state to first waypoint
SPACE 2: start trajectory
SPACE 3: hold current/final command for stick removal
SPACE 4: return to startup state, then release
```

For repeated viewing:

```bash
python apps/replay_trajectory.py \
  --interface eth0 \
  --domain-id 0 \
  --trajectory dual_hold_swing_v0 \
  --arm-sdk \
  --max-step-rad 0.003 \
  --loop
```

In loop mode, the third SPACE stops at the current command and holds there.

## Emergency Stop

Use the robot-side physical emergency stop when needed.

From the terminal, `Ctrl+C` asks the app to hold briefly, disable commands, and
release. It is software-level only, not a substitute for the physical stop.
