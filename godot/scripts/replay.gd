extends Node3D

@export var trajectory_path: String = ""
@export var speed: float = 1.0

var data: Array = []
var agents: Dictionary = {}
var idx: int = 0
@onready var camrig: Node = get_node_or_null("CameraRig")

func _ready() -> void:
    _apply_env_overrides()
    if trajectory_path == "":
        trajectory_path = _find_latest_trajectory()
    if trajectory_path == "":
        push_error("Replay: no trajectory found")
        return
    data = _load_jsonl(trajectory_path)
    _spawn_agents()
    _focus_camera()
    _notify_hud_reset()

func _process(delta: float) -> void:
    if data.size() == 0:
        return
    var steps: int = int(ceil(speed))
    while steps > 0 and idx < data.size():
        var e: Variant = data[idx]
        if typeof(e) == TYPE_DICTIONARY and e.get("type", "") == "step":
            var name: StringName = StringName(e.get("agent", ""))
            if agents.has(name):
                var a: Node = agents[name]
                var p: Array = e.get("pos", [0,0,0])
                a.global_transform.origin = Vector3(p[0], p[1], p[2])
                a.velocity = Vector3.ZERO
        idx += 1
        steps -= 1

func _spawn_agents() -> void:
    # Assume two agents in file
    var seen := {}
    for e in data:
        if typeof(e) == TYPE_DICTIONARY and e.get("type", "") == "step":
            var n: StringName = StringName(e.get("agent", ""))
            if not seen.has(n):
                seen[n] = true
                var ps := preload("res://scenes/Agent.tscn")
                var inst: Node = ps.instantiate()
                add_child(inst)
                inst.name = n
                if inst.has_method("set_replay_mode"):
                    inst.call("set_replay_mode", true)
                agents[n] = inst
        if seen.size() >= 2:
            break

func _focus_camera() -> void:
    if camrig == null:
        return
    # Focus on first agent if present
    for k in agents.keys():
        var a = agents[k]
        if camrig.has_method("set_target"):
            camrig.call("set_target", a)
            return

func _notify_hud_reset() -> void:
    var huds := get_tree().get_nodes_in_group("hud")
    if huds.size() > 0 and huds[0].has_method("on_episode_reset"):
        huds[0].call_deferred("on_episode_reset")

func _load_jsonl(path: String) -> Array:
    var arr: Array = []
    var f := FileAccess.open(path, FileAccess.READ)
    if f:
        while not f.eof_reached():
            var line := f.get_line()
            if line == "":
                continue
            var ln := line.strip_edges()
            if not ln.begins_with("{"):
                continue
            var json := JSON.new()
            var err := json.parse(ln)
            if err != OK:
                continue
            var obj = json.get_data()
            if typeof(obj) == TYPE_DICTIONARY:
                arr.append(obj)
        f.close()
    return arr

func _apply_env_overrides() -> void:
    var v = OS.get_environment("AI_REPLAY_PATH")
    if v != "":
        trajectory_path = v

func _find_latest_trajectory() -> String:
    var dir := DirAccess.open("user://trajectories")
    if dir == null:
        return ""
    dir.list_dir_begin()
    var best := ""
    while true:
        var fn := dir.get_next()
        if fn == "":
            break
        if dir.current_is_dir():
            continue
        if fn.ends_with(".jsonl"):
            best = "user://trajectories/" + fn
    dir.list_dir_end()
    return best
