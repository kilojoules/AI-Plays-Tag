extends Node

# Bridges Godot agents with the Python WebSocket server.
# Usage: add this node to the main scene and set `client_path` to an rl_client.gd node.

@export var client_path: NodePath
@export var control_ai_for: Array[StringName] = ["Agent1"]  # names of agents to control via AI
@export var control_all_agents: bool = true                  # if true, control all agents in scene
@export var training_mode: bool = false
@export var ai_is_it: bool = true  # when true, AI agent starts as 'seeker' each reset; false trains hider
@export var step_tick_interval: int = 3  # physics ticks per environment step
# Hide-and-Seek reward parameters
@export var distance_reward_scale: float = 0.1   # small shaping reward based on distance change
@export var seeker_time_penalty: float = -0.01   # per-step penalty for seeker
@export var runner_survival_bonus: float = 0.02  # per-step bonus for runner
@export var win_bonus: float = 10.0              # bonus on win; negative on loss
@export var high_ground_bonus: float = 0.1       # per-step bonus for runner when Y > 1.5
@export var jump_near_bonus: float = 0.5         # bonus for seeker jump near target
@export var max_steps_per_episode: int = 800     # safety cap (not used for termination)
@export var log_trajectories: bool = false
@export var legacy_act_fallback: bool = false    # send single-agent 'act' requests alongside act_batch

var client: Node = null
var agents: Array = []
var last_actions: Dictionary = {}
var tick: int = 0
@export var request_every_n_physics_ticks: int = 2
var gm: Node = null
var ai_agent: Node = null
var other_agent: Node = null
var last_obs: Dictionary = {}
var last_action: Dictionary = {}
var steps_in_ep: int = 0
var prev_distance: float = 0.0
var tag_latched: bool = false
var npc_node: Node = null
var tag_attacker: Node = null
var tag_target: Node = null
var log_file: FileAccess = null
var episode_id: int = 0
var time_elapsed: float = 0.0
@export var time_limit_sec: float = 10.0

func _ready() -> void:
    _apply_env_overrides()
    client = get_node_or_null(client_path)
    if client == null:
        push_warning("RL Env: client not found; AI control disabled")
        return
    client.connect("message", Callable(self, "_on_client_message"))
    _gather_agents()
    if control_all_agents:
        control_ai_for.clear()
        for a in agents:
            control_ai_for.append(a.name)
    # Switch specified agents to AI mode
    for a in agents:
        if a.name in control_ai_for:
            a.control_mode = "ai"
    gm = _get_game_manager()
    if gm:
        gm.connect("tag_event", Callable(self, "_on_tag_event"))
        if gm.has_signal("timeout_event"):
            gm.connect("timeout_event", Callable(self, "_on_timeout_event"))
    _select_ai_and_other()
    if training_mode:
        _env_reset()
    # Request initial observation/action loop
    call_deferred("_kick")

func _gather_agents() -> void:
    agents = get_tree().get_nodes_in_group("agents")

func _kick() -> void:
    if client and client.connected:
        _request_actions()

func _physics_process(_delta: float) -> void:
    # Apply any latest actions to AI-controlled agents
    for a in agents:
        if a.name in control_ai_for and last_actions.has(a.name):
            var act: Vector3 = last_actions[a.name]
            # Map action x,z in [-1,1] to move input
            a.ai_move_input = Vector2(act.x, act.z).clamp(Vector2(-1,-1), Vector2(1,1))
            a.ai_jump = act.y
    tick += 1
    if request_every_n_physics_ticks > 0 and (tick % request_every_n_physics_ticks) == 0:
        _request_actions()
    # Environment step logic for training
    if training_mode and ai_agent != null and other_agent != null:
        time_elapsed += _delta
        if (tick % max(1, step_tick_interval)) == 0 and last_action.size() > 0:
            # Currently supports two controlled agents best; uses pair distance.
            var names: Array = control_ai_for.duplicate()
            var trans_array: Array = []
            var cur_distance: float = 0.0
            if ai_agent and other_agent:
                cur_distance = ai_agent.global_transform.origin.distance_to(other_agent.global_transform.origin)
            var progress: float = prev_distance - cur_distance  # positive when seeker closes distance
            var done: bool = false
            var give_tag_bonus: bool = tag_latched
            steps_in_ep += 1
            # Termination only when hider is found or timer expires
            if time_elapsed >= time_limit_sec:
                done = true
            for n in names:
                var a: Node = _find_agent_by_name(n)
                if a == null:
                    continue
                var obs_next: Array = _pack_obs(a)
                var obs_prev: Array = last_obs.get(n, obs_next)
                var act_prev: Array = last_action.get(n, [0.0,0.0,0.0])
                var is_seeker: bool = a.is_it
                var rew: float = 0.0
                if is_seeker:
                    # Seekers: close distance and small time penalty
                    rew = progress * distance_reward_scale + seeker_time_penalty
                    # Bonus for jumping near the target to encourage dynamic play
                    var near_dist: float = cur_distance
                    var jumped: bool = false
                    if last_action.has(n):
                        var la: Array = last_action[n]
                        if la.size() >= 3:
                            jumped = float(la[2]) > 0.5
                    if jumped and near_dist <= 3.0:
                        rew += jump_near_bonus
                else:
                    # Runners: increase distance and survive
                    rew = (-progress) * distance_reward_scale + runner_survival_bonus
                    # High ground bonus by height
                    if a.global_transform.origin.y > 1.5:
                        rew += high_ground_bonus
                if time_elapsed >= time_limit_sec:
                    # timeout: runner wins, seeker loses
                    if is_seeker:
                        rew -= abs(win_bonus)
                    else:
                        rew += abs(win_bonus)
                if give_tag_bonus:
                    if tag_attacker == a:
                        rew += win_bonus
                    if tag_target == a:
                        rew -= abs(win_bonus)
                trans_array.append({
                    "obs": obs_prev,
                    "action": act_prev,
                    "reward": rew,
                    "next_obs": obs_next,
                    "done": done or give_tag_bonus,
                    "info": {"agent": str(n)}
                })
                # Update last obs for next step
                last_obs[n] = obs_next
                _log_step(a)
            prev_distance = cur_distance
            # Send batched transitions
            if trans_array.size() > 0:
                _send_transition_batch(trans_array)
            if done or give_tag_bonus:
                tag_latched = false
                tag_attacker = null
                tag_target = null
                time_elapsed = 0.0
                _env_reset()
                return
            # Request next actions with new observations
            if client and client.connected:
                var payload: Dictionary = {}
                for n in names:
                    var a2: Node = _find_agent_by_name(n)
                    if a2:
                        payload[n] = last_obs.get(n, _pack_obs(a2))
                if payload.size() > 0:
                    client.send_json({"type": "act_batch", "obs": payload})

func _on_client_message(msg: Dictionary) -> void:
    var t: String = str(msg.get("type", ""))
    if t == "action_batch":
        var acts: Dictionary = msg.get("actions", {})
        for k in acts.keys():
            var arr: Array = acts[k]
            var jv: float = (arr[2] if arr.size() > 2 else 0.0)
            last_actions[k] = Vector3(arr[0], jv, arr[1])
            last_action[k] = [arr[0], arr[1], jv]
    elif t == "action":
        # Single action fallback for one-agent control
        var arr: Array = msg.get("action", [0.0, 0.0, 0.0])
        if control_ai_for.size() > 0:
            var name0: StringName = control_ai_for[0]
            var jv2: float = (arr[2] if arr.size() > 2 else 0.0)
            last_actions[name0] = Vector3(arr[0], jv2, arr[1])
            last_action[name0] = [arr[0], arr[1], jv2]

func _request_actions() -> void:
    # Send observations for AI-controlled agents and request actions.
    var payload: Dictionary = {}
    for a in agents:
        if a.name in control_ai_for:
            var obs_arr: Array = _pack_obs(a)
            payload[a.name] = obs_arr
            last_obs[a.name] = obs_arr
    if training_mode and prev_distance == 0.0 and ai_agent and other_agent:
        prev_distance = ai_agent.global_transform.origin.distance_to(other_agent.global_transform.origin)
    if payload.size() == 0:
        return
    if client and client.connected:
        # Prefer batch API if server supports it; otherwise fallback per-agent 'act'
        client.send_json({"type": "act_batch", "obs": payload})
        if legacy_act_fallback and control_ai_for.size() > 0:
            var agent0: Node = _find_agent_by_name(control_ai_for[0])
            if agent0:
                client.send_json({"type": "act", "obs": payload[agent0.name]})

func _pack_obs(a: Node) -> Array:
    # Observation mix: normalized position/velocity, relative opponent data, role flags,
    # forward vector, and ray-cast vision (distance + agent mask per ray).
    var obs: Array = []

    var apos: Vector3 = a.global_transform.origin
    var avel: Vector3 = a.velocity
    var other = _find_other_agent(a)
    var opos: Vector3 = Vector3.ZERO
    var ovel: Vector3 = Vector3.ZERO
    var other_is_it: float = 0.0
    if other:
        opos = other.global_transform.origin
        ovel = other.velocity
        var other_flag = other.get("is_it")
        if typeof(other_flag) == TYPE_BOOL:
            other_is_it = 1.0 if other_flag else 0.0

    var arena_half: float = 15.0
    var max_speed: float = 10.0

    obs.append(apos.x / arena_half)
    obs.append(apos.z / arena_half)
    obs.append(avel.x / max_speed)
    obs.append(avel.z / max_speed)
    obs.append((opos.x - apos.x) / arena_half)
    obs.append((opos.z - apos.z) / arena_half)
    obs.append((ovel.x - avel.x) / max_speed)
    obs.append((ovel.z - avel.z) / max_speed)
    var self_is_it: float = 0.0
    var self_flag = a.get("is_it")
    if typeof(self_flag) == TYPE_BOOL:
        self_is_it = 1.0 if self_flag else 0.0
    obs.append(self_is_it)
    obs.append(other_is_it)

    var forward: Vector3 = -a.global_transform.basis.z
    if forward.length() > 0.0001:
        forward = forward.normalized()
    obs.append(forward.x)
    obs.append(forward.z)

    var vc: Object = a.get_node_or_null("VisionCaster")
    var expected_rays: int = 36
    if vc:
        var rays_val = vc.get("rays")
        var rays_type = typeof(rays_val)
        if rays_type == TYPE_INT or rays_type == TYPE_FLOAT:
            expected_rays = int(rays_val)
    if vc and vc.has_method("get_distances") and vc.has_method("get_types"):
        var dists: PackedFloat32Array = vc.call("get_distances")
        var types: Array = vc.call("get_types")
        var ray_count: int = dists.size()
        if ray_count == 0:
            var rv = vc.get("rays")
            var rv_type = typeof(rv)
            if rv_type == TYPE_INT or rv_type == TYPE_FLOAT:
                ray_count = int(rv)
        expected_rays = max(expected_rays, ray_count)
        var max_dist: float = 1.0
        var md_val = vc.get("max_distance")
        var md_type = typeof(md_val)
        if md_type == TYPE_FLOAT or md_type == TYPE_INT:
            max_dist = max(0.001, float(md_val))
        for i in range(ray_count):
            var dist_norm: float = clamp(float(dists[i]) / max_dist, 0.0, 1.0)
            obs.append(dist_norm)
            var t: String = "none"
            if i < types.size():
                t = str(types[i])
            obs.append(1.0 if t == "agent" else 0.0)
        if ray_count < expected_rays:
            for _i in range(expected_rays - ray_count):
                obs.append(1.0)
                obs.append(0.0)
    else:
        for _i in range(expected_rays):
            obs.append(1.0)
            obs.append(0.0)

    return obs

func _find_other_agent(a: Node) -> Node:
    for x in agents:
        if x != a:
            return x
    return null

func _apply_env_overrides() -> void:
    # Allow shell-driven overrides for headless training
    var v
    v = OS.get_environment("AI_TRAINING_MODE")
    if v != "":
        var low = v.to_lower()
        training_mode = (low == "1" or low == "true" or low == "yes")
    v = OS.get_environment("AI_IS_IT")
    if v != "":
        var low2 = v.to_lower()
        ai_is_it = (low2 == "1" or low2 == "true" or low2 == "yes")
    v = OS.get_environment("AI_CONTROL_ALL_AGENTS")
    if v != "":
        var low3 = v.to_lower()
        control_all_agents = (low3 == "1" or low3 == "true" or low3 == "yes")
        _ensure_control_ai_subset()
    v = OS.get_environment("AI_DISTANCE_REWARD_SCALE")
    if v != "":
        distance_reward_scale = float(v)
    v = OS.get_environment("AI_LEGACY_ACT_FALLBACK")
    if v != "":
        var low4 = v.to_lower()
        legacy_act_fallback = (low4 == "1" or low4 == "true" or low4 == "yes")
    v = OS.get_environment("AI_SEEKER_TIME_PENALTY")
    if v != "":
        seeker_time_penalty = float(v)
    v = OS.get_environment("AI_WIN_BONUS")
    if v != "":
        win_bonus = float(v)
    v = OS.get_environment("AI_STEP_TICK_INTERVAL")
    if v != "":
        step_tick_interval = int(v)
    v = OS.get_environment("AI_MAX_STEPS_PER_EPISODE")
    if v != "":
        max_steps_per_episode = int(v)

func _get_game_manager() -> Node:
    var nodes = get_tree().get_nodes_in_group("game_manager")
    if nodes.size() > 0:
        return nodes[0]
    return null

func _select_ai_and_other() -> void:
    _ensure_control_ai_subset()
    ai_agent = null
    other_agent = null
    # Prefer first two from control_ai_for for pairing
    var first: StringName = ""
    var second: StringName = ""
    if control_ai_for.size() >= 1:
        first = control_ai_for[0]
    if control_ai_for.size() >= 2:
        second = control_ai_for[1]
    ai_agent = _find_agent_by_name(first)
    if second != "":
        other_agent = _find_agent_by_name(second)
    # Fallbacks if not found
    if ai_agent == null and agents.size() > 0:
        ai_agent = agents[0]
    if other_agent == null:
        for a in agents:
            if a != ai_agent:
                other_agent = a
                break

func _env_reset() -> void:
    last_obs.clear()
    last_action.clear()
    steps_in_ep = 0
    tag_latched = false
    tag_attacker = null
    tag_target = null
    time_elapsed = 0.0
    _close_log()
    episode_id += 1
    _open_log()
    if gm and ai_agent:
        var it_node: Node = ai_agent
        if not ai_is_it and other_agent:
            it_node = other_agent
        gm.reset_round(it_node)
    # Attach or configure NPC behavior for the non-AI agent
    _setup_npc_for_other()
    _update_npc_mode()
    # Recompute distance
    if ai_agent and other_agent:
        prev_distance = ai_agent.global_transform.origin.distance_to(other_agent.global_transform.origin)
    # Request initial action
    if client and client.connected and ai_agent:
        var obs = _pack_obs(ai_agent)
        last_obs[ai_agent.name] = obs
        client.send_json({"type": "act", "obs": obs})
    # Notify HUD
    var huds = get_tree().get_nodes_in_group("hud")
    if huds.size() > 0 and huds[0].has_method("on_episode_reset"):
        huds[0].call_deferred("on_episode_reset")

func _on_tag_event(attacker: Node, target: Node) -> void:
    # End the episode for all controlled agents when a tag occurs
    tag_latched = true
    tag_attacker = attacker
    tag_target = target
    _log_event({"type":"tag","attacker": attacker.name, "target": target.name})
    call_deferred("_update_npc_mode")

func _on_timeout_event() -> void:
    # Hider escape: handled by time-based termination in step loop; log for completeness
    _log_event({"type":"timeout"})

func _find_agent_by_name(n: StringName) -> Node:
    for a in agents:
        if a.name == n:
            return a
    return null

func _send_transition(obs_prev: Array, act_prev: Array, reward: float, obs_next: Array, done: bool) -> void:
    if client and client.connected:
        client.send_json({
            "type": "transition",
            "obs": obs_prev,
            "action": act_prev,
            "reward": reward,
            "next_obs": obs_next,
            "done": done,
            "info": {}
        })

func _send_transition_batch(transitions: Array) -> void:
    if client and client.connected:
        client.send_json({
            "type": "transition_batch",
            "transitions": transitions
        })

func _setup_npc_for_other() -> void:
    # Ensure the non-AI agent moves: wander if AI is chaser; chase AI if AI is runner.
    if other_agent == null or control_ai_for.size() > 1:
        # When controlling multiple agents, disable NPC controller.
        return
    # Clean previous npc node
    if npc_node and is_instance_valid(npc_node):
        npc_node.queue_free()
        npc_node = null
    var npc_script = load("res://scripts/npc_controller.gd")
    npc_node = npc_script.new()
    other_agent.add_child(npc_node)
    other_agent.control_mode = "ai"
    _update_npc_mode()

func _update_npc_mode() -> void:
    if npc_node == null or other_agent == null:
        return
    if control_ai_for.size() > 1:
        return
    if not (is_instance_valid(npc_node) and is_instance_valid(other_agent)):
        return
    var other_flag = other_agent.get("is_it")
    var other_is_it = typeof(other_flag) == TYPE_BOOL and other_flag
    var next_mode := "wander"
    if other_is_it and ai_agent:
        next_mode = "chase"
        var path: NodePath = other_agent.get_path_to(ai_agent)
        npc_node.target_path = path
    elif npc_node.target_path != NodePath(""):
        npc_node.target_path = NodePath("")
    npc_node.mode = next_mode
    if npc_node.has_method("refresh_behavior"):
        npc_node.call_deferred("refresh_behavior")

func _ensure_control_ai_subset() -> void:
    if control_all_agents:
        return
    if control_ai_for.size() == 0 and agents.size() > 0:
        control_ai_for.append(StringName(agents[0].name))
    if control_ai_for.size() > 1:
        var first: StringName = control_ai_for[0]
        control_ai_for.clear()
        control_ai_for.append(first)

func _open_log() -> void:
    if not log_trajectories:
        var v = OS.get_environment("AI_LOG_TRAJECTORIES")
        if v != "":
            var low = v.to_lower()
            log_trajectories = (low == "1" or low == "true" or low == "yes")
    if not log_trajectories:
        return
    var dir = DirAccess.open("user://")
    if dir:
        dir.make_dir_recursive("trajectories")
    var fname = "user://trajectories/ep_%05d.jsonl" % episode_id
    log_file = FileAccess.open(fname, FileAccess.WRITE)
    _log_event({"type":"episode_start","episode": episode_id})

func _close_log() -> void:
    if log_file:
        _log_event({"type":"episode_end","episode": episode_id})
        log_file.close()
        log_file = null

func _log_event(e: Dictionary) -> void:
    if log_file:
        e["ts"] = Time.get_ticks_msec()
        log_file.store_line(JSON.stringify(e))

func _log_step(a: Node) -> void:
    if not log_file:
        return
    var vc: Node = a.get_node_or_null("VisionCaster")
    var pos_y: float = a.global_transform.origin.y
    if pos_y < 0.0:
        var anomaly = {
            "type": "physics_anomaly",
            "episode": episode_id,
            "agent": a.name,
            "position": [a.global_transform.origin.x, pos_y, a.global_transform.origin.z],
            "velocity": [a.velocity.x, a.velocity.y, a.velocity.z]
        }
        if a is CharacterBody3D:
            anomaly["is_on_floor"] = (a as CharacterBody3D).is_on_floor()
        _log_event(anomaly)
    var entry = {
        "type": "step",
        "episode": episode_id,
        "agent": a.name,
        "pos": [a.global_transform.origin.x, a.global_transform.origin.y, a.global_transform.origin.z],
        "vel": [a.velocity.x, a.velocity.y, a.velocity.z],
        "is_it": a.is_it,
    }
    if vc:
        var dists: PackedFloat32Array = vc.call("get_distances")
        var types: Array = vc.call("get_types")
        var darr: Array = []
        var tarr: Array = []
        if dists:
            for v in dists:
                darr.append(v)
        if types:
            for t in types:
                tarr.append(t)
        entry["dists"] = darr
        entry["types"] = tarr
    _log_event(entry)
