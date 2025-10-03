# WebSocket Protocol Reference

The Godot environment (`RLEnv`) and the Python trainer (`trainer/server.py`) communicate over a lightweight JSON protocol.

## Connection
- Server listens on `ws://127.0.0.1:8765` (`trainer/server.py`).
- Godot clients should send JSON payloads; plain HTTP connections will be rejected.

## Messages from Godot → Python

### `act_batch`
```json
{
  "type": "act_batch",
  "obs": {
    "Agent1": [/* observation vector */],
    "Agent2": [/* observation vector */]
  }
}
```
- Required when requesting actions for one or more agents in the same physics tick.
- Server responds with `action_batch`.

### `act`
```json
{
  "type": "act",
  "obs": [/* observation vector */]
}
```
- Optional legacy fallback (disabled by default via `legacy_act_fallback`).
- Server responds with `action`.

### `transition_batch`
```json
{
  "type": "transition_batch",
  "transitions": [
    {
      "obs": [/* previous observation */],
      "action": [/* action taken */],
      "reward": float,
      "next_obs": [/* optional */],
      "done": bool,
      "info": {"agent": "Agent1"}
    }
  ]
}
```
- Preferred method for streaming experience tuples; server buffers and triggers PPO updates when batches reach the configured threshold.

### `transition`
- Single-transition version of the above. Kept for compatibility.

## Messages from Python → Godot

### `action_batch`
```json
{
  "type": "action_batch",
  "actions": {
    "Agent1": [/* move_x, move_z, jump */]
  }
}
```

### `action`
```json
{
  "type": "action",
  "action": [/* move_x, move_z, jump */]
}
```
- Reply to legacy `act` requests.

### `echo`
- Diagnostic response when an unknown message type is received.

## Error Handling
- Invalid observation payloads result in a safe `[0,0,0]` action and an `info.error` flag (`invalid_obs`).
- Policy dimension changes reset the learning buffers automatically (see `Trainer.ensure_policy`).

## Tips
- Always send batched messages once multiple agents are controlled to avoid action staleness.
- Enable `AI_LOG_TRAJECTORIES=1` during training to line up WebSocket activity with on-disk experience dumps for debugging.
