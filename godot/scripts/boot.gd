extends Node

func _ready() -> void:
    var run_tests := OS.get_environment("AI_RUN_TESTS")
    if run_tests != "":
        var low := run_tests.to_lower()
        if low == "1" or low == "true" or low == "yes":
            var test_scene_env := OS.get_environment("AI_TEST_SCENE")
            var test_scene := "res://tests/TestRunner.tscn"
            if test_scene_env.strip_edges() != "":
                test_scene = test_scene_env
            call_deferred("_change", test_scene)
            return
    var scene := OS.get_environment("AI_BOOT_SCENE")
    if scene != "":
        call_deferred("_change", scene)

func _change(scene: String) -> void:
    var ok := get_tree().change_scene_to_file(scene)
    if ok != OK:
        push_error("Failed to change scene to %s: %s" % [scene, ok])
