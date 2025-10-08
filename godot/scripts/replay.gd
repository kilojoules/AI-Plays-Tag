extends Node3D
const DataPaths = preload("res://scripts/data_paths.gd")

@export var trajectory_path: String = ""
@export var speed: float = 1.0
@export var render_quality: String = "high"

var data: Array = []
var agents: Dictionary = {}
var idx: int = 0
@onready var camrig: Node = get_node_or_null("CameraRig")

func _ready() -> void:
    _apply_env_overrides()
    _apply_quality_preset()
    if trajectory_path == "":
        trajectory_path = _find_latest_trajectory()
    trajectory_path = _resolve_trajectory_path(trajectory_path)
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
    v = OS.get_environment("AI_RENDER_QUALITY")
    if v != "":
        render_quality = v
    render_quality = _normalize_quality(render_quality)

func _normalize_quality(val: String) -> String:
    var lower := val.to_lower()
    if lower == "high":
        return "high"
    return "low"

func _apply_quality_preset() -> void:
    var preset := _normalize_quality(render_quality)
    render_quality = preset
    if preset == "high":
        _apply_high_quality()
        print("[Replay] Render quality preset: high")
    else:
        _apply_low_quality()
        print("[Replay] Render quality preset: low")

func _apply_low_quality() -> void:
    var world_env: WorldEnvironment = get_node_or_null("WorldEnvironment")
    if world_env and world_env.environment:
        var env: Environment = world_env.environment
        env.set("ambient_light_energy", 1.5)
        env.set("auto_exposure_enabled", false)
        env.set("ssao_enabled", false)
        env.set("ssr_enabled", false)
        env.set("sdfgi_enabled", false)
        env.set("glow_enabled", false)
        env.set("volumetric_fog_enabled", false)
    _configure_light("DirectionalLight3D", false, false)
    _configure_light("FillLight", false, false)
    _configure_light("BackSpot", false, false)
    var probe: ReflectionProbe = get_node_or_null("ReflectionProbe")
    if probe:
        probe.visible = false
        probe.intensity = 0.0

func _apply_high_quality() -> void:
    var world_env: WorldEnvironment = get_node_or_null("WorldEnvironment")
    if world_env and world_env.environment:
        var env: Environment = world_env.environment
        env.set("ambient_light_energy", 1.0)
        env.set("auto_exposure_enabled", true)
        env.set("ssao_enabled", true)
        env.set("ssr_enabled", true)
        env.set("sdfgi_enabled", true)
        env.set("glow_enabled", true)
        env.set("volumetric_fog_enabled", true)
    _configure_light("DirectionalLight3D", true, true)
    _configure_light("FillLight", true, true)
    _configure_light("BackSpot", true, true)
    var dir := get_node_or_null("DirectionalLight3D") as DirectionalLight3D
    if dir:
        dir.light_energy = 2.4
        dir.light_indirect_energy = 1.1
        dir.light_specular = 1.2
    var fill := get_node_or_null("FillLight") as OmniLight3D
    if fill:
        fill.light_energy = 1.1
        fill.light_specular = 1.1
    var back := get_node_or_null("BackSpot") as SpotLight3D
    if back:
        back.light_energy = 1.8
        back.light_indirect_energy = 0.85
    var probe: ReflectionProbe = get_node_or_null("ReflectionProbe")
    if probe:
        probe.visible = true
        probe.intensity = 1.1

func _configure_light(node_name: String, enable: bool, shadow_enable: bool) -> void:
    var n = get_node_or_null(node_name)
    if n == null:
        return
    if n is Light3D:
        var light := n as Light3D
        light.visible = enable
        light.shadow_enabled = enable and shadow_enable

func _find_latest_trajectory() -> String:
    var search_dirs: Array[String] = []
    search_dirs.append(DataPaths.trajectories_dir())
    var legacy := DataPaths.legacy_trajectory_dirs()
    for dir_path in legacy:
        search_dirs.append(dir_path)
    for dir_path in search_dirs:
        var dir := DirAccess.open(dir_path)
        if dir == null:
            continue
        dir.list_dir_begin()
        var best_name := ""
        while true:
            var fn := dir.get_next()
            if fn == "":
                break
            if dir.current_is_dir():
                continue
            if fn.ends_with(".jsonl") and (best_name == "" or fn > best_name):
                best_name = fn
        dir.list_dir_end()
        if best_name != "":
            return dir_path.path_join(best_name)
    return ""

func _resolve_trajectory_path(path: String) -> String:
    var trimmed := path.strip_edges()
    if trimmed == "":
        return ""
    if FileAccess.file_exists(trimmed):
        return trimmed
    if trimmed.begins_with("res://") or trimmed.begins_with("user://"):
        var global := ProjectSettings.globalize_path(trimmed)
        if FileAccess.file_exists(global):
            return global
    var workspace := DataPaths.trajectories_dir().path_join(trimmed)
    if FileAccess.file_exists(workspace):
        return workspace
    for legacy_dir in DataPaths.legacy_trajectory_dirs():
        var candidate := legacy_dir.path_join(trimmed)
        if FileAccess.file_exists(candidate):
            return candidate
    return trimmed
