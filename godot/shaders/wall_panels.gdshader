shader_type spatial;
render_mode cull_back, specular_schlick_ggx;

uniform vec3 base_color : source_color = vec3(0.10, 0.11, 0.15);
uniform vec3 accent_color : source_color = vec3(0.14, 0.16, 0.22);
uniform vec3 seam_color : source_color = vec3(0.05, 0.06, 0.08);
uniform float panel_scale : hint_range(0.5, 10.0) = 3.25;
uniform float seam_width : hint_range(0.001, 0.2) = 0.035;
uniform float gradient_strength : hint_range(0.0, 1.0) = 0.5;
uniform float sheen_strength : hint_range(0.0, 1.0) = 0.3;
uniform float noise_strength : hint_range(0.0, 0.2) = 0.025;

float hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p.x + p.y) * 43758.5453123);
}

void fragment() {
    vec2 uv = UV;
    vec2 panel_uv = uv * vec2(panel_scale, 1.5 * panel_scale);
    vec2 cell = floor(panel_uv);
    float rnd = hash(cell);

    vec2 local = abs(fract(panel_uv) - 0.5);
    float seam_v = smoothstep(0.5 - seam_width, 0.5, local.x);
    float seam_h = smoothstep(0.5 - seam_width * 0.5, 0.5, local.y);
    float seam_mix = 1.0 - max(seam_v, seam_h * 0.4);

    float gradient = pow(clamp(UV.y, 0.0, 1.0), 1.5);

    float noise = (hash(panel_uv) - 0.5) * noise_strength;

    vec3 color = mix(seam_color, base_color, seam_mix);
    color = mix(color, accent_color, rnd * 0.35 + gradient * gradient_strength);
    color += sheen_strength * gradient * vec3(0.12, 0.14, 0.18);
    color += noise;

    ALBEDO = color;
    ROUGHNESS = clamp(0.35 + rnd * 0.2 + gradient * 0.1, 0.08, 0.85);
    METALLIC = 0.05;
    SPECULAR = 0.35;

    float seam_emission = pow(1.0 - seam_mix, 3.0);
    EMISSION = vec3(0.06, 0.15, 0.28) * seam_emission * 0.6;

}
