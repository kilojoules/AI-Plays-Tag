extends CanvasLayer

@export var hider_agent_name: StringName = "Agent1"
@export var seeker_agent_name: StringName = "Agent2"

var hider: Node = null
var seeker: Node = null
var gm: Node = null
var score: int = 0
var episode: int = 0
var t: float = 0.0

func _ready() -> void:
    add_to_group("hud")
    gm = _get_game_manager()
    if gm:
        gm.connect("tag_event", Callable(self, "_on_found"))
        if gm.has_signal("timeout_event"):
            gm.connect("timeout_event", Callable(self, "_on_timeout"))
    _find_agents()

func _process(delta: float) -> void:
    t += delta
    var info_label: Label = $Panel/InfoLabel
    if info_label:
        var gm := _get_game_manager()
        var tl := 0.0
        if gm and ("time_left" in gm):
            tl = float(gm.time_left)
        info_label.text = "Episode: %d  Time: %.1fs  Time Left: %.1fs" % [episode, t, tl]
    _set_pane_targets()

func _on_found(attacker: Node, target: Node) -> void:
    # Seeker wins this round
    var label: Label = $Panel/ScoreLabel
    if label:
        label.text = "Seeker Wins"

func _on_timeout() -> void:
    # Hider escapes (time out)
    var label: Label = $Panel/ScoreLabel
    if label:
        label.text = "Hider Escapes!"

func on_episode_reset() -> void:
    episode += 1
    t = 0.0
    var label: Label = $Panel/ScoreLabel
    if label:
        label.text = ""

func _find_agents() -> void:
    var nodes := get_tree().get_nodes_in_group("agents")
    for n in nodes:
        if n.name == hider_agent_name:
            hider = n
        elif n.name == seeker_agent_name:
            seeker = n

func _set_pane_targets() -> void:
    if has_node("HiderPane"):
        var hp = $HiderPane
        if hp and hp.has_method("set") and hider:
            hp.set("agent_name", hider.name)
    if has_node("SeekerPane"):
        var sp = $SeekerPane
        if sp and sp.has_method("set") and seeker:
            sp.set("agent_name", seeker.name)

func _get_game_manager() -> Node:
    var nodes := get_tree().get_nodes_in_group("game_manager")
    if nodes.size() > 0:
        return nodes[0]
    return null
