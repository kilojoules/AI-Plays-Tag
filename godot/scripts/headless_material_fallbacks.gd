extends Node

class_name HeadlessMaterialFallbacks

const RenderEnv = preload("res://scripts/render_environment.gd")

static func apply(root: Node) -> void:
    if root == null:
        return
    if not RenderEnv.is_headless():
        return
    _apply_arena_materials(root)
    _disable_environment_effects(root)

static func _apply_arena_materials(root: Node) -> void:
    var targets := {
        "Floor/MeshInstance3D": Color(0.20, 0.22, 0.26),
        "WallN/MeshN": Color(0.12, 0.14, 0.18),
        "WallS/MeshS": Color(0.12, 0.14, 0.18),
        "WallE/MeshE": Color(0.12, 0.14, 0.18),
        "WallW/MeshW": Color(0.12, 0.14, 0.18),
    }
    for path in targets.keys():
        var mesh := root.get_node_or_null(path)
        if mesh and mesh is MeshInstance3D:
            _assign_standard_material(mesh, targets[path])
    _apply_agent_materials(root)

static func _assign_standard_material(mesh: MeshInstance3D, color: Color) -> void:
    if mesh.mesh == null:
        return
    var mat := StandardMaterial3D.new()
    mat.albedo_color = color
    mat.metallic = 0.0
    mat.roughness = 0.7
    mat.emission_enabled = false
    mesh.material_override = null
    mesh.set_surface_override_material(0, mat)

static func _apply_agent_materials(root: Node) -> void:
    var agents: Array = []
    # Prefer direct children with agent script for deterministic coverage.
    for child in root.get_children():
        if child == null:
            continue
        var script = child.get_script()
        if script and script.resource_path.ends_with("agent.gd"):
            agents.append(child)
    if agents.is_empty():
        _gather_agents(root, agents)
    for agent in agents:
        _standardize_agent(agent)

static func _gather_agents(node: Node, acc: Array) -> void:
    if node.is_in_group("agents"):
        acc.append(node)
    for child in node.get_children():
        if child is Node:
            _gather_agents(child, acc)

static func _standardize_agent(agent: Node) -> void:
    var mesh_names := ["MeshInstance3D", "FootL", "FootR"]
    for name in mesh_names:
        var node := agent.get_node_or_null(name)
        if node and node is MeshInstance3D:
            _assign_standard_material(node, Color(0.6, 0.6, 0.65))
    if agent.has_method("set_it"):
        agent.call("set_it", agent.get("is_it"))

static func _disable_environment_effects(root: Node) -> void:
    var env_node := root.get_node_or_null("WorldEnvironment")
    if env_node == null:
        return
    if not (env_node is WorldEnvironment):
        return
    var env := (env_node as WorldEnvironment).environment
    if env == null:
        return
    env.ssao_enabled = false
    env.ssr_enabled = false
    env.sdfgi_enabled = false
    env.glow_enabled = false
