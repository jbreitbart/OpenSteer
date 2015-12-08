PROJECT(opensteer)

# set the include path
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../include/)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)

# get all the source files
# this is dirty, if you add a new file to one of the directories, you need to rerun cmake(!) by hand(!!)
AUX_SOURCE_DIRECTORY(../src             SRC)
AUX_SOURCE_DIRECTORY(../src/Graphics    SRC_GRAPHICS)
AUX_SOURCE_DIRECTORY(../plugins         SRC_PLUGINS)
AUX_SOURCE_DIRECTORY(../kapaga          SRC_KAPAGA)
AUX_SOURCE_DIRECTORY(../kapaga/posix    SRC_KAPAGA_POSIX)

# set the compiler macros
ADD_DEFINITIONS(-DOPENSTEER)
ADD_DEFINITIONS(-DUSEOpenGL)

# if you don't want the full compiler output, remove the following line
#SET(CMAKE_VERBOSE_MAKEFILE ON)

#add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -W -pedantic -O2 -g)

ADD_EXECUTABLE(opensteer ${SRC} ${SRC_GRAPHICS} ${SRC_PLUGINS} ${SRC_KAPAGA} ${SRC_KAPAGA_POSIX})

TARGET_LINK_LIBRARIES(opensteer glut GLU GL pthread)

