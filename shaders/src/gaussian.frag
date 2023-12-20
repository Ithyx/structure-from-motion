#version 450

layout(set = 3, binding = 1) uniform ColorData {
    vec3 color;
} u_Color;

layout(location = 0) out vec4 f_Color;

void main() {
    f_Color = vec4(u_Color.color, 1.0);
}

