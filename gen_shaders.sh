mkdir -p shaders/gen
glslc shaders/src/gaussian.vert -o shaders/gen/gaussian.vert.spirv
glslc shaders/src/gaussian.frag -o shaders/gen/gaussian.frag.spirv
