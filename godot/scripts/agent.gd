extends CharacterBody3D

const RenderEnv = preload("res://scripts/render_environment.gd")
const AGENT_SHADER_PATH := "res://materials/agent_body.tres"

@export var speed: float = 8.0
@export var acceleration: float = 20.0
@export var deceleration: float = 30.0
@export var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")
@export var jump_velocity: float = 5.5

@export_enum("human", "ai") var control_mode: String = "human"

var ai_move_input: Vector2 = Vector2.ZERO
var ai_jump: float = 0.0
var is_it: bool = false
var immune_until: float = 0.0
var replay_freeze: bool = false
var dead: bool = false
var spawn_jitter_until: float = 0.0
var spawn_jitter_vec: Vector2 = Vector2.ZERO

# Jump feel helpers
var coyote_time_sec: float = 0.1
var coyote_timer: float = 0.0
var was_on_floor: bool = false
var exit_on_fall: bool = false

@onready var tag_area: Area3D = $TagArea
@onready var mesh: MeshInstance3D = $MeshInstance3D
@onready var eyes: Node3D = $Eyes

func _ready() -> void:
    add_to_group("agents")
    if tag_area:
        tag_area.body_entered.connect(_on_tag_area_body_entered)
    _maybe_apply_runtime_materials()
    # Neutral default; GameManager will set role colors via set_it()
    _apply_color(Color(0.7, 0.7, 0.7))
    # Improve ground contact
    floor_snap_length = 0.5
    floor_stop_on_slope = true
    floor_block_on_wall = true
    safe_margin = 0.01
    floor_max_angle = 0.785398  # 45 degrees
    # Strict CI mode: exit when a fall below the floor is detected
    var v := OS.get_environment("AI_EXIT_ON_FALL")
    if v != "":
        var low := v.to_lower()
        exit_on_fall = (low == "1" or low == "true" or low == "yes")

func _physics_process(delta: float) -> void:
    if replay_freeze or dead:
        return
    # Coyote time tracking
    var on_floor := is_on_floor()
    if on_floor:
        coyote_timer = coyote_time_sec
    else:
        coyote_timer = max(0.0, coyote_timer - delta)
    was_on_floor = on_floor
    var input_vec := Vector2.ZERO
    if control_mode == "human":
        input_vec = Input.get_vector("move_left", "move_right", "move_forward", "move_back")
    else:
        input_vec = ai_move_input
        # Add brief random jitter after spawn to avoid immediate straight runs
        var now := float(Time.get_ticks_msec()) / 1000.0
        if now < spawn_jitter_until:
            if spawn_jitter_vec == Vector2.ZERO:
                _refresh_spawn_jitter()
            # Bias toward jitter vector the first frames
            input_vec = input_vec.lerp(spawn_jitter_vec, 0.7)

    var direction := (transform.basis * Vector3(input_vec.x, 0, input_vec.y)).normalized()

    var target_hvel := direction * speed
    var hvel := Vector3(velocity.x, 0, velocity.z)
    if direction.length() > 0.01:
        hvel = hvel.move_toward(target_hvel, acceleration * delta)
    else:
        hvel = hvel.move_toward(Vector3.ZERO, deceleration * delta)

    velocity.x = hvel.x
    velocity.z = hvel.z

    if not on_floor:
        velocity.y -= gravity * delta
    else:
        # Maintain a gentle downward bias so slide-with-snap keeps contact.
        var denom: float = max(delta, 0.0001)
        velocity.y = -floor_snap_length / denom * 0.1
        if control_mode == "human" and Input.is_action_just_pressed("jump"):
            if on_floor or coyote_timer > 0.0:
                velocity.y = jump_velocity
                coyote_timer = 0.0
        elif control_mode == "ai":
            # Treat ai_jump > 0.5 as jump signal
            if ai_jump > 0.5 and (on_floor or coyote_timer > 0.0):
                velocity.y = jump_velocity
                coyote_timer = 0.0

    move_and_slide()
    _enforce_arena_bounds()
    # Fail-safe: don't allow falling through floor
    if global_transform.origin.y < -0.5:
        _recover_from_fall()
    # Collision feedback: simple hook when sliding collisions occur
    if get_slide_collision_count() > 0:
        _on_collision_feedback()

    # Update eyes to look in movement direction
    var look_dir := Vector3(velocity.x, 0.0, velocity.z)
    if look_dir.length() < 0.05:
        look_dir = -transform.basis.z
    if eyes and eyes.has_method("set_look_direction"):
        eyes.call("set_look_direction", look_dir.normalized())
    # Simple squash-and-stretch while airborne (visual only)
    var target_scale := Vector3.ONE
    if not on_floor:
        target_scale = Vector3(1.0, 1.3, 1.0)
    if mesh and mesh is MeshInstance3D:
        mesh.scale = (mesh.scale).lerp(target_scale, clamp(10.0 * delta, 0.0, 1.0))

func set_replay_mode(active: bool) -> void:
    replay_freeze = active
    set_physics_process(not active)

func _on_tag_area_body_entered(body: Node) -> void:
    if body == self:
        return
    if body is CharacterBody3D and body.is_in_group("agents"):
        var gm := _get_game_manager()
        if gm:
            # Hide-and-seek: notify manager when seeker finds a hider
            if gm.has_method("_on_hider_found"):
                gm._on_hider_found(self, body)
            else:
                gm.try_tag(self, body)

func set_it(active: bool) -> void:
    is_it = active
    # Visual feedback via simple color modulation
    if active:
        _apply_color(Color(1.0, 0.3, 0.3))  # seeker (red)
    else:
        _apply_color(Color(0.3, 0.3, 1.0))  # runner (blue)
    # Update overhead label
    var lbl := get_node_or_null("RoleLabel")
    if lbl and lbl is Label3D:
        (lbl as Label3D).text = ("SEEKER" if active else "HIDER")

func _on_death() -> void:
    # Play a simple death effect: fade/scale and disable controls + physics
    dead = true
    replay_freeze = true
    set_physics_process(false)
    set_process(false)
    ai_move_input = Vector2.ZERO
    ai_jump = 0.0
    # Disable collisions
    set_collision_layer(0)
    set_collision_mask(0)
    if tag_area:
        tag_area.monitoring = false
        tag_area.set_deferred("monitoring", false)
    # Tween scale down for a quick "poof"
    var tw := create_tween()
    tw.tween_property(self, "scale", Vector3(0.1, 0.1, 0.1), 0.35).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_IN)

func revive() -> void:
    # Reset state after death for new episode
    dead = false
    replay_freeze = false
    set_physics_process(true)
    set_process(true)
    set_collision_layer(1)
    set_collision_mask(1)
    if mesh and mesh is MeshInstance3D:
        mesh.scale = Vector3.ONE
    if tag_area:
        tag_area.monitoring = true
        tag_area.set_deferred("monitoring", true)
    on_spawn()

func on_spawn() -> void:
    var now := float(Time.get_ticks_msec()) / 1000.0
    spawn_jitter_until = now + 0.8
    _refresh_spawn_jitter()

func _refresh_spawn_jitter() -> void:
    var rng := RandomNumberGenerator.new()
    rng.randomize()
    var v := Vector2(rng.randf_range(-1,1), rng.randf_range(-1,1))
    if v.length() < 0.1:
        v = Vector2(1,0)
    spawn_jitter_vec = v.normalized()

func _recover_from_fall() -> void:
    var space_state := get_world_3d().direct_space_state
    if space_state == null:
        return
    var origin := global_transform.origin
    origin.y = 5.0
    var query := PhysicsRayQueryParameters3D.create(origin, origin + Vector3.DOWN * 20.0)
    query.exclude = [self]
    var result := space_state.intersect_ray(query)
    if result:
        global_transform.origin = result.position + Vector3.UP * 0.1
        velocity = Vector3.ZERO
        push_warning("Agent '%s' recovered from fall." % name)
    else:
        global_transform.origin = Vector3.ZERO
        velocity = Vector3.ZERO
        push_error("Agent '%s' fell with no ground detected." % name)
    if exit_on_fall:
        get_tree().quit(1)

func _enforce_arena_bounds() -> void:
    var gm := _get_game_manager()
    var limit := 14.5
    if gm and ("arena_half_size" in gm):
        limit = float(gm.arena_half_size) - 0.3
    limit = min(limit, 6.5)
    var pos := global_transform.origin
    var clamped := false
    if pos.x > limit:
        pos.x = limit
        velocity.x = min(velocity.x, 0.0)
        clamped = true
    elif pos.x < -limit:
        pos.x = -limit
        velocity.x = max(velocity.x, 0.0)
        clamped = true
    if pos.z > limit:
        pos.z = limit
        velocity.z = min(velocity.z, 0.0)
        clamped = true
    elif pos.z < -limit:
        pos.z = -limit
        velocity.z = max(velocity.z, 0.0)
        clamped = true
    if clamped:
        global_transform.origin = pos

func _on_collision_feedback() -> void:
    # Placeholder for visual/sfx feedback when bumping walls
    # Could emit particles or briefly tint the mesh.
    if mesh and mesh is MeshInstance3D:
        var m := mesh.get_surface_override_material(0)
        if m is StandardMaterial3D:
            var orig: Color = (m as StandardMaterial3D).albedo_color
            (m as StandardMaterial3D).albedo_color = orig.lightened(0.15)
            await get_tree().process_frame
            (m as StandardMaterial3D).albedo_color = orig

func _apply_color(c: Color) -> void:
    # Apply to main mesh and any child meshes (e.g., feet) so the whole agent reflects the role color.
    var meshes: Array = []
    if mesh:
        meshes.append(mesh)
    for child in get_children():
        if child is MeshInstance3D and child != mesh:
            meshes.append(child)
    for m in meshes:
        var mat: Material = (m as MeshInstance3D).get_surface_override_material(0)
        if mat == null:
            var new_mat := StandardMaterial3D.new()
            new_mat.roughness = 0.4
            new_mat.metallic = 0.0
            (m as MeshInstance3D).set_surface_override_material(0, new_mat)
            mat = new_mat
        if mat is StandardMaterial3D:
            var sm := (mat as StandardMaterial3D)
            sm.albedo_color = c
            # Subtle emissive to help readability with glow
            sm.emission_enabled = true
            sm.emission = c * 0.35
            sm.emission_energy_multiplier = 1.2
        elif mat is ShaderMaterial:
            _tint_shader_material(mat as ShaderMaterial, c)
    # Tint trail particles to match role
    var tp := get_node_or_null("TrailParticles")
    if tp and tp is GPUParticles3D:
        var pm := (tp as GPUParticles3D).process_material
        if pm and pm is ParticleProcessMaterial:
            (pm as ParticleProcessMaterial).color = Color(c.r, c.g, c.b, 0.8)
    var hl := get_node_or_null("HaloLight")
    if hl and hl is OmniLight3D:
        (hl as OmniLight3D).light_color = c

func _maybe_apply_runtime_materials() -> void:
    if RenderEnv.is_headless():
        return
    var shader_res: Material = load(AGENT_SHADER_PATH)
    if shader_res == null:
        return
    if shader_res is ShaderMaterial:
        _assign_material_to_mesh(mesh, (shader_res as ShaderMaterial).duplicate())
        _assign_material_to_mesh(get_node_or_null("FootL"), (shader_res as ShaderMaterial).duplicate())
        _assign_material_to_mesh(get_node_or_null("FootR"), (shader_res as ShaderMaterial).duplicate())
    else:
        _assign_material_to_mesh(mesh, shader_res.duplicate())
        _assign_material_to_mesh(get_node_or_null("FootL"), shader_res.duplicate())
        _assign_material_to_mesh(get_node_or_null("FootR"), shader_res.duplicate())

func _assign_material_to_mesh(node: Node, material: Material) -> void:
    if node == null:
        return
    if not (node is MeshInstance3D):
        return
    var mesh_instance := node as MeshInstance3D
    mesh_instance.material_override = null
    mesh_instance.set_surface_override_material(0, material)

func _tint_shader_material(mat: ShaderMaterial, color: Color) -> void:
    var base := Vector3(color.r, color.g, color.b)
    var accent := color.lerp(Color(1, 1, 1), 0.35)
    var rim := color.lerp(Color(1, 1, 1), 0.5)
    mat.set_shader_parameter("base_color", base)
    mat.set_shader_parameter("accent_color", Vector3(accent.r, accent.g, accent.b))
    mat.set_shader_parameter("rim_color", Vector3(rim.r, rim.g, rim.b))
    if mat.shader and mat.shader.has_uniform("emission_strength"):
        mat.set_shader_parameter("emission_strength", 1.8)

# (ground clamp removed; replaced with thicker floor collider)

func _get_game_manager() -> Node:
    var nodes := get_tree().get_nodes_in_group("game_manager")
    if nodes.size() > 0:
        return nodes[0]
    return null
