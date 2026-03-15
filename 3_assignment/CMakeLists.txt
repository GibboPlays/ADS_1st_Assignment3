cmake_minimum_required(VERSION 3.10)
project(ParticleFilterProject)

set(CMAKE_CXX_STANDARD 17)

# 1. Cerca PCL (Point Cloud Library)
find_package(PCL REQUIRED)

# 2. Includi le directory (cartella locale 'include')
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 3. Definisci i file sorgente (percorso relativo alla posizione di questo file)
set(SOURCE_FILES 
    src/main.cpp 
    src/particle_filter.cpp 
    src/Renderer.cpp
)

# 4. Crea l'eseguibile
add_executable(particle_filter ${SOURCE_FILES})

# 5. Collega le librerie PCL
target_link_libraries(particle_filter ${PCL_LIBRARIES})