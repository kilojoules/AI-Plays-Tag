extends Node3D

@export var duration_sec: float = 15.0
@export var seeker_mode: StringName = "chase"   # "chase" or "wander"
@export var hider_mode: StringName = "wander"   # "wander" or "chase"

var t: float = 0.0
var gm: Node = null
var seeker: Node = null
var hider: Node = null
var hide_hud: bool = false

func _ready() -> void:
    _apply_env_overrides()
    gm = _get_game_manager()
    _find_agents()
    _attach_npc()
    _enable_recorder()
    _maybe_toggle_hud()

func _process(delta: float) -> void:
    t += delta
    if duration_sec > 0.0 and t >= duration_sec:
        # Finish cleanly for automated capture runs
        get_tree().quit()

func _find_agents() -> void:
    var list := get_tree().get_nodes_in_group("agents")
    if list.size() == 0:
        return
    # Default roles follow GameManager's setup (first is seeker, second hider)
    seeker = list[0]
    if list.size() > 1:
        hider = list[1]

func _attach_npc() -> void:
    if seeker:
        var npc_script = load("res://scripts/npc_controller.gd")
        var npc = npc_script.new()
        npc.mode = String(seeker_mode)
        if npc.mode == "chase" and hider:
            npc.target_path = seeker.get_path_to(hider)
        seeker.add_child(npc)
        seeker.control_mode = "ai"
    if hider:
        var npc_script2 = load("res://scripts/npc_controller.gd")
        var npc2 = npc_script2.new()
        npc2.mode = String(hider_mode)
        if npc2.mode == "chase" and seeker:
            npc2.target_path = hider.get_path_to(seeker)
        hider.add_child(npc2)
        hider.control_mode = "ai"

func _enable_recorder() -> void:
    var rec := get_node_or_null("Recorder")
    if rec and rec.has_method("set"):
        rec.set("enabled", true)

func _get_game_manager() -> Node:
    var nodes := get_tree().get_nodes_in_group("game_manager")
    if nodes.size() > 0:
        return nodes[0]
    return null

func _apply_env_overrides() -> void:
    var v = OS.get_environment("AI_DEMO_DURATION_SEC")
    if v != "":
        duration_sec = float(v)
    v = OS.get_environment("AI_DEMO_HIDE_HUD")
    if v != "":
        var low := v.to_lower()
        hide_hud = (low == "1" or low == "true" or low == "yes")

func _maybe_toggle_hud() -> void:
    if not hide_hud:
        return
    var hud_nodes := get_tree().get_nodes_in_group("hud")
    for h in hud_nodes:
        if h is CanvasLayer:
            (h as CanvasLayer).visible = false
