#version 450

layout(location = 0) in vec2 vertex;
layout(location = 1) in vec2 tcoord;

layout(location = 0) out vec2 ftcoord;
layout(location = 1) out vec2 fpos;

layout(push_constant) uniform View
{
    vec2 size;
} view;

void main()
{
    ftcoord     = tcoord;
    fpos        = vertex;
    gl_Position = vec4(2.0 * vertex.x / view.size.x - 1.0, 2.0 * vertex.y / view.size.y - 1.0, 0.0, 1.0);
}
