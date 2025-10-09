extends Node3D

const HeadlessFallbacks = preload("res://scripts/headless_material_fallbacks.gd")

const TEST_DURATION_SEC := 20.0
var agents: Array = []
var failures: int = 0
var world_scene: Node3D = null
var log_file: FileAccess = null
var log_frame: int = 0
var stuck_frames: Dictionary = {}
var separation_frames: int = 0

func _ready() -> void:
    randomize()
    var packed := preload("res://scenes/Main.tscn")
    world_scene = packed.instantiate()
    _strip_runtime_nodes(world_scene)
    add_child(world_scene)
    await get_tree().process_frame
    HeadlessFallbacks.apply(world_scene)
    agents = get_tree().get_nodes_in_group("agents")
    if agents.is_empty():
        push_error("[StressTest] No agents found in instantiated scene.")
        get_tree().quit(1)
        return
    var gm := world_scene.get_node_or_null("GameManager")
    if gm and gm.has_method("reset_round"):
        gm.call_deferred("reset_round", agents[0])
    for agent in agents:
        if agent is CharacterBody3D:
            (agent as CharacterBody3D).control_mode = "ai"
        _ensure_npc(agent)
        _teleport_agent_randomly(agent)
    _open_log()
    get_tree().create_timer(TEST_DURATION_SEC).timeout.connect(func():
        _close_log()
        print("[StressTest] Completed with %d failures." % failures)
        get_tree().quit(failures)
    )

func _physics_process(_delta: float) -> void:
    for agent in agents:
        if not is_instance_valid(agent):
            continue
        if agent is Node3D and (agent as Node3D).global_transform.origin.y < -0.1:
            failures += 1
            push_error("Agent '%s' fell through the floor." % agent.name)
            _teleport_agent_randomly(agent)
            stuck_frames[agent] = 0
            continue
        if agent is CharacterBody3D:
            var body := agent as CharacterBody3D
            var speed := body.velocity.length()
            if body.is_on_floor() and speed < 0.4:
                var frames: int = int(stuck_frames.get(agent, 0)) + 1
                stuck_frames[agent] = frames
                if frames > 180:
                    _nudge_agent(body)
            else:
                stuck_frames[agent] = 0
    _maybe_reduce_separation()
    _log_state()

func _teleport_agent_randomly(agent: Variant) -> void:
    if not (agent is Node3D):
        return
    var node := agent as Node3D
    var x := randf_range(-6.0, 6.0)
    var z := randf_range(-6.0, 6.0)
    var y := randf_range(1.2, 3.5)
    node.global_transform.origin = Vector3(x, y, z)
    if agent is CharacterBody3D:
        (agent as CharacterBody3D).velocity = Vector3.ZERO
        (agent as CharacterBody3D).ai_jump = 0.0

func _strip_runtime_nodes(root: Node) -> void:
    var removable := ["RLClient", "RLEnv"]
    for name in removable:
        var node := root.get_node_or_null(name)
        if node:
            root.remove_child(node)
            node.free()
    var recorder := root.get_node_or_null("Recorder")
    if recorder:
        root.remove_child(recorder)
        recorder.free()

func _ensure_npc(agent: Node) -> void:
    if not (agent is Node3D):
        return
    if agent.get_node_or_null("StressNPC"):
        return
    var npc_script := load("res://scripts/npc_controller.gd")
    var npc = npc_script.new()
    npc.name = "StressNPC"
    npc.change_dir_interval_ticks = 90
    npc.jump_interval_ticks = 180
    if agent == agents[0]:
        npc.mode = "wander"
    elif agents.size() > 0 and agent != agents[0]:
        npc.mode = "chase"
    npc.target_path = (agent as Node3D).get_path_to(agents[0])
    (agent as Node3D).add_child(npc)
    npc.refresh_behavior()

func _open_log() -> void:
    var dir := DirAccess.open("user://")
    if dir:
        dir.make_dir_recursive("stress_logs")
    var ts := Time.get_ticks_msec()
    var path := "user://stress_logs/run_%010d.jsonl" % ts
    log_file = FileAccess.open(path, FileAccess.WRITE)
    log_frame = 0
    if log_file:
        log_file.store_line(JSON.stringify({"type":"start","timestamp": ts}))

func _close_log() -> void:
    if log_file:
        log_file.store_line(JSON.stringify({"type":"end","frame": log_frame}))
        log_file.close()
        log_file = null

func _log_state() -> void:
    if log_file == null:
        return
    var entry := {
        "type": "frame",
        "frame": log_frame,
        "agents": []
    }
    for agent in agents:
        if not is_instance_valid(agent):
            continue
        var node := agent as Node3D
        var data := {
            "name": String(agent.name),
            "pos": [node.global_transform.origin.x, node.global_transform.origin.y, node.global_transform.origin.z]
        }
        if agent is CharacterBody3D:
            data["vel"] = [agent.velocity.x, agent.velocity.y, agent.velocity.z]
            data["is_on_floor"] = (agent as CharacterBody3D).is_on_floor()
        entry["agents"].append(data)
    log_file.store_line(JSON.stringify(entry))
    log_frame += 1

func _exit_tree() -> void:
    _close_log()

func _nudge_agent(agent: CharacterBody3D) -> void:
    agent.velocity = Vector3.ZERO
    var impulse := Vector3(randf_range(-2.0, 2.0), 0.0, randf_range(-2.0, 2.0))
    agent.ai_move_input = Vector2(impulse.x, impulse.z).clamp(Vector2(-1,-1), Vector2(1,1))
    agent.ai_jump = 0.0
    var npc := agent.get_node_or_null("StressNPC")
    if npc and npc.has_method("refresh_behavior"):
        npc.call("refresh_behavior")
    stuck_frames[agent] = 0

func _maybe_reduce_separation() -> void:
    if agents.size() < 2:
        return
    var a: Node = agents[0]
    var b: Node = agents[1]
    if not (is_instance_valid(a) and is_instance_valid(b)):
        return
    var dist: float = (a.global_transform.origin - b.global_transform.origin).length()
    if dist > 8.0:
        separation_frames += 1
    else:
        separation_frames = max(separation_frames - 1, 0)
    if separation_frames > 120:
        separation_frames = 0
        _teleport_near(b, a.global_transform.origin)

func _teleport_near(agent: Node, center: Vector3) -> void:
    if not (agent is CharacterBody3D):
        return
    var char := agent as CharacterBody3D
    var offset := Vector3(randf_range(-2.0, 2.0), randf_range(1.0, 2.0), randf_range(-2.0, 2.0))
    char.global_transform.origin = center + offset
    char.velocity = Vector3.ZERO
    char.ai_jump = 0.0
    var npc := char.get_node_or_null("StressNPC")
    if npc and npc.has_method("refresh_behavior"):
        npc.call("refresh_behavior")
