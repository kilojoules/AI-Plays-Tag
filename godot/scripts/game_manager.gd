extends Node

const RenderEnv = preload("res://scripts/render_environment.gd")

@export var post_tag_immunity_sec: float = 1.5
@export var arena_half_size: float = 8.0
@export var time_limit_sec: float = 10.0

var agents: Array = []
# Hide-and-Seek roles (fixed per episode)
var seeker_agent: Node = null
var hider_agent: Node = null
@onready var camera_rig: Node3D = null
var camera_index: int = 0

signal tag_event(attacker, target)  # emitted when hider is found by seeker
signal timeout_event()              # emitted when episode time expires (hider escapes)

var start_time: float = 0.0
var time_left: float = 0.0
var episode_active: bool = false

func _ready() -> void:
    add_to_group("game_manager")
    _apply_env_overrides()
    agents = get_tree().get_nodes_in_group("agents")
    if agents.size() == 0:
        push_warning("No agents found in scene.")
        return
    _apply_physics_materials()
    _apply_runtime_materials()
    # Default roles before first reset
    seeker_agent = agents[0]
    hider_agent = (agents[1] if agents.size() > 1 else agents[0])
    _apply_roles()
    # Setup camera rig
    camera_rig = get_node_or_null("../CameraRig")
    if camera_rig and camera_rig.has_method("set_target"):
        camera_rig.call_deferred("set_target", seeker_agent)
        # Ensure follow mode is active for visibility
        if camera_rig.has_method("set"):
            camera_rig.set("fixed_camera", false)
            camera_rig.set("focus_group", "agents")
    _restart_timer(true)

func _process(_delta: float) -> void:
    if Input.is_action_just_pressed("switch_agent"):
        _switch_camera_target()
    if episode_active:
        # Update countdown
        var now := float(Time.get_ticks_msec()) / 1000.0
        time_left = max(0.0, time_limit_sec - (now - start_time))
        if time_left <= 0.0:
            # Hider survives: hider wins, seeker loses.
            _end_episode(false)

func try_tag(attacker: Node, target: Node) -> void:
    # Backward compatibility: delegate to _on_hider_found
    _on_hider_found(attacker, target)

func _on_hider_found(attacker: Node, target: Node) -> void:
    # Only seeker can find hider, and only while episode is active
    if not episode_active:
        return
    if attacker != seeker_agent or target != hider_agent:
        return
    var now := float(Time.get_ticks_msec()) / 1000.0
    if hider_agent and hider_agent.immune_until > now:
        return
    emit_signal("tag_event", attacker, target)
    # Seeker wins, hider loses
    _end_episode(true)

func _apply_roles() -> void:
    for a in agents:
        a.set_it(false)
    if seeker_agent:
        seeker_agent.set_it(true)
        var now := float(Time.get_ticks_msec()) / 1000.0
        seeker_agent.immune_until = now + post_tag_immunity_sec
    if camera_rig and camera_rig.has_method("set_target") and seeker_agent:
        camera_rig.call_deferred("set_target", seeker_agent)

func _switch_camera_target() -> void:
    if agents.size() == 0:
        return
    camera_index = (camera_index + 1) % agents.size()
    var target: Node = agents[camera_index]
    if camera_rig and camera_rig.has_method("set_target"):
        camera_rig.call("set_target", target)

func _apply_env_overrides() -> void:
    var v := OS.get_environment("AI_TIME_LIMIT_SEC")
    if v != "":
        time_limit_sec = float(v)

func reset_round(seeker: Node) -> void:
    # Randomize positions and reset states; assign roles for this episode.
    if agents.size() == 0:
        agents = get_tree().get_nodes_in_group("agents")
    var rng := RandomNumberGenerator.new()
    rng.randomize()
    var margin := 2.0
    var min_sep := 6.0
    var positions: Array = []
    for a in agents:
        var x := rng.randf_range(-arena_half_size + margin, arena_half_size - margin)
        var z := rng.randf_range(-arena_half_size + margin, arena_half_size - margin)
        # Ensure separation from previously placed agents
        var tries := 0
        while tries < 20:
            var ok := true
            for p in positions:
                if Vector2(x, z).distance_to(Vector2(p.x, p.z)) < min_sep:
                    ok = false
                    break
            if ok:
                break
            x = rng.randf_range(-arena_half_size + margin, arena_half_size - margin)
            z = rng.randf_range(-arena_half_size + margin, arena_half_size - margin)
            tries += 1
        # Spawn slightly above floor for reliable ground contact
        a.global_transform.origin = Vector3(x, 1.0, z)
        a.velocity = Vector3.ZERO
        a.immune_until = 0.0
        a.set_it(false)
        positions.append(Vector3(x, 0.0, z))
        if a.has_method("on_spawn"):
            a.call("on_spawn")
        # Re-enable physics/controls in case of prior death
        if a.has_method("revive"):
            a.call("revive")
    if seeker == null:
        seeker = agents[0]
    seeker_agent = seeker
    # Choose hider as any other agent (prefer next in list)
    hider_agent = null
    for a2 in agents:
        if a2 != seeker_agent:
            hider_agent = a2
            break
    if hider_agent == null:
        hider_agent = seeker_agent
    _apply_roles()
    _restart_timer(true)

func _restart_timer(active: bool) -> void:
    start_time = float(Time.get_ticks_msec()) / 1000.0
    time_left = time_limit_sec
    episode_active = active

func _end_episode(seeker_wins: bool) -> void:
    episode_active = false
    time_left = 0.0
    # Play death/found animation on loser
    var loser: Node = (hider_agent if seeker_wins else seeker_agent)
    if loser and loser.has_method("_on_death"):
        loser.call("_on_death")
    if not seeker_wins:
        emit_signal("timeout_event")

func _apply_physics_materials() -> void:
    # Increase friction on floor and walls to reduce sliding.
    var pm := PhysicsMaterial.new()
    pm.friction = 1.0
    pm.bounce = 0.0
    var root := get_parent()
    if root == null:
        return
    var names := ["Floor", "WallN", "WallS", "WallE", "WallW"]
    for n in names:
        var node := root.get_node_or_null(n)
        if node and node is PhysicsBody3D:
            (node as PhysicsBody3D).physics_material_override = pm
    # Validate that critical static bodies exist and include collision shapes.
    for body_name in names:
        var body := root.get_node_or_null(body_name)
        assert(body != null, "Missing critical static body: %s" % body_name)
        assert(body is StaticBody3D, "%s should be a StaticBody3D" % body_name)
        var has_shape := false
        for child in body.get_children():
            if child is CollisionShape3D:
                has_shape = true
                break
        assert(has_shape, "Missing CollisionShape3D for: %s" % body_name)
    print_rich("[color=lightgreen][Physics][/color] Static colliders validated.")

func _apply_runtime_materials() -> void:
    if RenderEnv.is_headless():
        return
    var root := get_parent()
    if root == null:
        return
    var floor_mesh := root.get_node_or_null("Floor/MeshInstance3D")
    var floor_mat: Material = load("res://materials/floor_tiles.tres")
    if floor_mesh and floor_mesh is MeshInstance3D and floor_mat:
        (floor_mesh as MeshInstance3D).material_override = null
        (floor_mesh as MeshInstance3D).set_surface_override_material(0, floor_mat)
    var wall_mat: Material = load("res://materials/wall_panels.tres")
    var wall_paths := [
        "WallN/MeshN",
        "WallS/MeshS",
        "WallE/MeshE",
        "WallW/MeshW",
    ]
    if wall_mat:
        for path in wall_paths:
            var mesh := root.get_node_or_null(path)
            if mesh and mesh is MeshInstance3D:
                (mesh as MeshInstance3D).material_override = null
                (mesh as MeshInstance3D).set_surface_override_material(0, wall_mat)
    var env_node := root.get_node_or_null("WorldEnvironment")
    if env_node and env_node is WorldEnvironment:
        var env := (env_node as WorldEnvironment).environment
        if env:
            env.ssao_enabled = true
            env.ssr_enabled = true
            env.sdfgi_enabled = true
            env.glow_enabled = true
            env.volumetric_fog_enabled = true
