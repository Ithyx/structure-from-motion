#version 450

layout(location = 0) in vec3 v_Position;

layout(push_constant) uniform CameraData {
    mat4 viewProjection;
    vec4 worldPos;
}
pc_CameraData;

layout(set = 3, binding = 0) uniform ModelData { mat4 modelMatrix; }
u_ModelData;

void main() {
    mat4 transform = pc_CameraData.viewProjection * u_ModelData.modelMatrix;
    gl_Position = transform * vec4(v_Position, 1);
}

