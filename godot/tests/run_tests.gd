extends Node

const HeadlessFallbacks = preload("res://scripts/headless_material_fallbacks.gd")

# Simple headless test runner for Godot 4.
# Usage:
#   GODOT_BIN=/path/to/Godot --headless --script res://tests/run_tests.gd

var failures = 0

func _ready() -> void:
    print("[tests] Initializing test runner ...")
    await _run_all()
    get_tree().quit(failures)

func _get_main_scene_instance() -> Node:
    var ps := preload("res://scenes/Main.tscn")
    var inst = ps.instantiate()
    # Disable RL networking before adding to tree to avoid WebSocket connects
    var rlclient = inst.get_node_or_null("RLClient")
    if rlclient:
        rlclient.connect_on_ready = false
    var rlenv = inst.get_node_or_null("RLEnv")
    if rlenv:
        rlenv.training_mode = false
    get_tree().root.call_deferred("add_child", inst)
    await get_tree().process_frame
    await get_tree().physics_frame
    HeadlessFallbacks.apply(inst)
    return inst

func _assert(cond: bool, msg: String) -> void:
    if cond:
        print("[PASS] ", msg)
    else:
        failures += 1
        push_error("[FAIL] %s" % msg)

func _find_agents() -> Array:
    return get_tree().get_nodes_in_group("agents")

func _physics_steps(n: int) -> void:
    for i in range(n):
        await get_tree().physics_frame

func _process_steps(n: int) -> void:
    for i in range(n):
        await get_tree().process_frame

func _run_all() -> void:
    var world = await _get_main_scene_instance()
    await _process_steps(3)
    await _test_no_fall_through_floor()
    await _test_walls_block_agents()
    await _test_observation_vector(world)
    await _test_camera_scaling(world)
    await _test_npc_roles(world)
    world.queue_free()

func _test_no_fall_through_floor() -> void:
    print("[tests] floor thickness / grounding (10s, track min y)")
    var ag = _find_agents()
    _assert(ag.size() >= 1, "at least one agent present")
    if ag.size() == 0:
        return
    var a = ag[0]
    a.control_mode = "ai"
    a.global_transform.origin = Vector3(0, 5, 0)
    a.velocity = Vector3.ZERO
    var min_y := 9999.0
    var space_state: PhysicsDirectSpaceState3D = a.get_world_3d().direct_space_state
    if space_state == null:
        _assert(false, "physics space unavailable for ground check")
        return
    for i in range(600): # ~10 seconds
        await get_tree().physics_frame
        var y = a.global_transform.origin.y
        if y < min_y:
            min_y = y
        if i > 120:
            var from: Vector3 = a.global_transform.origin
            var to: Vector3 = from + Vector3.DOWN * 2.0
            var query := PhysicsRayQueryParameters3D.create(from, to)
            query.exclude = [a]
            var hit: Dictionary = space_state.intersect_ray(query)
            _assert(hit.size() > 0, "Agent should always find ground below it.")
            if hit:
                var ground_dist: float = from.distance_to(hit.position)
                _assert(ground_dist < 0.6, "Agent ground distance should be minimal, got %f" % ground_dist)
    _assert(min_y >= -0.1, "agent never dips below safety threshold (min_y>=-0.1), got %f" % min_y)
    _assert(a.is_on_floor(), "agent must report grounded state after settle")

func _test_walls_block_agents() -> void:
    print("[tests] perimeter walls block movement")
    var ag = _find_agents()
    _assert(ag.size() >= 1, "at least one agent present")
    if ag.size() == 0:
        return
    var a = ag[0]
    a.control_mode = "ai"
    # Push East (+X)
    a.global_transform.origin = Vector3(14.0, 1.0, 0.0)
    a.velocity = Vector3.ZERO
    a.ai_move_input = Vector2(1, 0)
    await _physics_steps(180)
    _assert(a.global_transform.origin.x <= 14.7, "east wall stops agent (x<=14.7), got %f" % a.global_transform.origin.x)
    # Push West (-X)
    a.global_transform.origin = Vector3(-14.0, 1.0, 0.0)
    a.velocity = Vector3.ZERO
    a.ai_move_input = Vector2(-1, 0)
    await _physics_steps(180)
    _assert(a.global_transform.origin.x >= -14.7, "west wall stops agent (x>=-14.7), got %f" % a.global_transform.origin.x)
    # Push North (-Z)
    a.global_transform.origin = Vector3(0.0, 1.0, -14.0)
    a.velocity = Vector3.ZERO
    a.ai_move_input = Vector2(0, -1)
    await _physics_steps(180)
    _assert(a.global_transform.origin.z >= -14.7, "north wall stops agent (z>=-14.7), got %f" % a.global_transform.origin.z)
    # Push South (+Z)
    a.global_transform.origin = Vector3(0.0, 1.0, 14.0)
    a.velocity = Vector3.ZERO
    a.ai_move_input = Vector2(0, 1)
    await _physics_steps(180)
    _assert(a.global_transform.origin.z <= 14.7, "south wall stops agent (z<=14.7), got %f" % a.global_transform.origin.z)

func _test_camera_scaling(world: Node) -> void:
    print("[tests] camera framing adjusts by distance")
    var rig: Node3D = world.get_node_or_null("CameraRig")
    _assert(rig != null, "CameraRig present")
    if rig == null:
        return
    rig.fixed_camera = false
    var agents = _find_agents()
    _assert(agents.size() >= 2, "at least two agents present for camera validation")
    if agents.size() < 2:
        return
    var a: Node3D = agents[0]
    var b: Node3D = agents[1]
    a.global_transform.origin = Vector3(-1.0, 1.0, 0.0)
    b.global_transform.origin = Vector3(1.0, 1.0, 0.0)
    a.velocity = Vector3.ZERO
    b.velocity = Vector3.ZERO
    await _process_steps(20)
    var center := (a.global_transform.origin + b.global_transform.origin) * 0.5
    var near_dist := rig.global_transform.origin.distance_to(center)
    var cam: Camera3D = rig.get_node("SpringArm3D/Camera3D")
    var near_fov := cam.fov
    a.global_transform.origin = Vector3(-12.0, 1.0, 0.0)
    b.global_transform.origin = Vector3(12.0, 1.0, 0.0)
    await _process_steps(40)
    center = (a.global_transform.origin + b.global_transform.origin) * 0.5
    var far_dist := rig.global_transform.origin.distance_to(center)
    var far_fov := cam.fov
    _assert(far_dist > near_dist + 1.0, "camera backs up when agents separate (%.2f -> %.2f)" % [near_dist, far_dist])
    _assert(far_fov >= near_fov - 0.1, "camera FOV widens for distant agents (%.2f -> %.2f)" % [near_fov, far_fov])

func _test_npc_roles(world: Node) -> void:
    print("[tests] npc behavior follows seeker/hider roles")
    var env: Node = world.get_node_or_null("RLEnv")
    _assert(env != null, "RLEnv present for NPC test")
    if env == null:
        return
    env.control_all_agents = false
    env.call("_ensure_control_ai_subset")
    env.training_mode = true
    env.ai_is_it = false
    env.call("_select_ai_and_other")
    env.call("_env_reset")
    await _process_steps(20)
    var other_agent: Node = env.get("other_agent")
    var ai_agent: Node = env.get("ai_agent")
    _assert(other_agent != null and ai_agent != null, "env resolved AI/other agents")
    if other_agent == null or ai_agent == null:
        return
    var npc = null
    for child in other_agent.get_children():
        if child.get_script() and child.get_script().resource_path.ends_with("npc_controller.gd"):
            npc = child
            break
    _assert(npc != null, "NPC attached to non-AI agent")
    if npc == null:
        return
    _assert(npc.mode == "chase", "NPC chases AI when other agent is seeker")
    # Simulate role swap: AI becomes seeker after tagging event
    other_agent.set("is_it", false)
    ai_agent.set("is_it", true)
    env.call("_on_tag_event", other_agent, ai_agent)
    await _process_steps(10)
    _assert(npc.mode == "wander", "NPC switches to wander after AI becomes seeker")
    _assert(npc.target_path == NodePath(""), "NPC clears chase target when wandering")

func _test_observation_vector(world: Node) -> void:
    print("[tests] observation vector sanity")
    var rlenv: Node = world.get_node_or_null("RLEnv")
    _assert(rlenv != null, "RLEnv node present")
    if rlenv == null:
        return
    var ag = _find_agents()
    _assert(ag.size() >= 1, "at least one agent present for observation")
    if ag.size() == 0:
        return
    var obs_variant = rlenv.call("_pack_obs", ag[0])
    _assert(typeof(obs_variant) == TYPE_ARRAY, "_pack_obs returns Array")
    if typeof(obs_variant) != TYPE_ARRAY:
        return
    var obs: Array = obs_variant
    var base_count = 12
    var ray_pairs = 36
    var expected_len = base_count + ray_pairs * 2
    _assert(obs.size() == expected_len, "observation length matches expected (%d), got %d" % [expected_len, obs.size()])
    if obs.size() != expected_len:
        return
    for i in range(ray_pairs):
        var idx = base_count + i * 2
        var dist_val = float(obs[idx])
        _assert(dist_val >= 0.0 and dist_val <= 1.0001, "ray distance normalized (0-1), got %f" % dist_val)
        var mask_val = float(obs[idx + 1])
        _assert(mask_val == 0.0 or mask_val == 1.0, "ray agent mask is binary, got %f" % mask_val)
