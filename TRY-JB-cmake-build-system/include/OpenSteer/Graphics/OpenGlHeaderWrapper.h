/**
    @file

    Platform dependent includes for @c gl.h, @c glu.h, and @c glut.h.

 */
#ifndef OPENSTEER_OPENGLHEADERWRAPPER_H
#define OPENSTEER_OPENGLHEADERWRAPPER_H


// @todoRevisit conditionalization on operating system.
// Mac OS X
#if __APPLE__ && __MACH__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #ifndef HAVE_NO_GLUT
        #include <GLUT/glut.h>
    #endif
#else
    // Windows
    #ifdef _MSC_VER
        #include <windows.h>
    #endif
    // Windows and Unix, Linux, etc. 
    #include <GL/gl.h>
    #include <GL/glu.h>
    #ifndef HAVE_NO_GLUT
        #include <GL/glut.h>
    #endif
#endif


#endif // OPENSTEER_OPENGLHEADERWRAPPER_H
