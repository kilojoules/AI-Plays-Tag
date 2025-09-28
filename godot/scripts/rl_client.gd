extends Node

# WebSocket client stub for Phase 2. Not used by default.
@export var url: String = "ws://127.0.0.1:8765"
@export var connect_on_ready: bool = true
var ws := WebSocketPeer.new()
var connected: bool = false

func _ready() -> void:
    if connect_on_ready:
        connect_ws()

func connect_ws() -> void:
    var err := ws.connect_to_url(url)
    if err != OK:
        push_error("WebSocket connect failed: %s" % err)

func _process(_delta: float) -> void:
    if ws.get_ready_state() == WebSocketPeer.STATE_CONNECTING:
        ws.poll()
    elif ws.get_ready_state() == WebSocketPeer.STATE_OPEN:
        if not connected:
            connected = true
        ws.poll()
        while ws.get_available_packet_count() > 0:
            var pkt := ws.get_packet().get_string_from_utf8()
            _handle_message(pkt)

func send_json(d: Dictionary) -> void:
    if ws.get_ready_state() == WebSocketPeer.STATE_OPEN:
        var s := JSON.stringify(d)
        ws.send_text(s)

func _handle_message(s: String) -> void:
    var result = JSON.parse_string(s)
    if typeof(result) == TYPE_DICTIONARY:
        emit_signal("message", result)

signal message(msg)
