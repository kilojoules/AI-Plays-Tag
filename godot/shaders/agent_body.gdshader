shader_type spatial;
render_mode cull_back, specular_schlick_ggx;

varying vec3 world_pos;

uniform vec3 base_color : source_color = vec3(0.12, 0.2, 0.35);
uniform vec3 accent_color : source_color = vec3(0.35, 0.72, 0.9);
uniform vec3 rim_color : source_color = vec3(0.45, 0.9, 1.0);
uniform float accent_height : hint_range(0.1, 4.0) = 1.8;
uniform float accent_softness : hint_range(0.01, 2.0) = 0.8;
uniform float rim_strength : hint_range(0.0, 2.0) = 0.6;
uniform float rim_power : hint_range(0.5, 8.0) = 2.5;
uniform float emission_strength : hint_range(0.0, 8.0) = 1.5;
uniform float stripe_scale : hint_range(0.2, 10.0) = 3.0;
uniform float stripe_contrast : hint_range(0.0, 1.0) = 0.35;

void vertex() {
    vec4 wp = MODEL_MATRIX * vec4(VERTEX, 1.0);
    world_pos = wp.xyz;
}

void fragment() {
    float height_mask = clamp((world_pos.y + accent_height * 0.25) / accent_height, 0.0, 1.0);
    height_mask = pow(height_mask, accent_softness);

    vec2 planar = world_pos.xz * stripe_scale;
    float stripes = sin(planar.x * 3.14159);
    stripes = mix(0.5, 1.0, smoothstep(0.0, 1.0, stripes * 0.5 + 0.5));

    vec3 albedo = mix(base_color, accent_color, height_mask);
    albedo = mix(albedo, albedo * 1.25, stripe_contrast * (stripes - 0.5));

    vec3 N = normalize(NORMAL);
    vec3 V = normalize(-VIEW);
    float rim = pow(clamp(1.0 - dot(N, V), 0.0, 1.0), rim_power) * rim_strength;

    ALBEDO = albedo;
    ROUGHNESS = mix(0.12, 0.35, height_mask);
    METALLIC = 0.05;
    SPECULAR = 0.35;

    EMISSION = rim_color * rim * emission_strength;
}
