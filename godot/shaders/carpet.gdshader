shader_type spatial;
// Enable lighting and shadows so the floor receives and shows depth cues.
render_mode cull_back;

uniform vec3 color_a : source_color = vec3(0.90, 0.45, 0.45);
uniform vec3 color_b : source_color = vec3(0.96, 0.82, 0.78);
uniform float scale = 4.0; // tiles per meter
uniform float line_thickness = 0.05;
uniform vec3 line_color : source_color = vec3(0.15, 0.15, 0.15);

float checker(vec2 p) {
    vec2 ip = floor(p);
    return mod(ip.x + ip.y, 2.0);
}

void fragment() {
    vec2 p = UV * scale * 10.0; // PlaneMesh UV is 0..1; multiply to get size in meters
    float c = checker(p);
    vec3 base = mix(color_a, color_b, c);
    // Thin stitch lines
    float lx = abs(fract(p.x) - 0.5);
    float ly = abs(fract(p.y) - 0.5);
    float line = smoothstep(0.5 - line_thickness, 0.5, max(lx, ly));
    vec3 col = mix(line_color, base, line);
    ALBEDO = col;
}
