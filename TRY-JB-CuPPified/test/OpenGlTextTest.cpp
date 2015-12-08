// Include EXIT_SUCCESS
#include <cstdlib>

// Include std::cerr, std::cout, std::endl
#include <iostream>

// Include std::string
#include <string>

// Include std::ostrstream
#include <sstream>


// Include OpenGL, glu, glut.
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::translatioMatrix
#include "OpenSteer/MatrixUtilities.h"

// Include OpenSteer::Graphics::OpenGlRenderer
#include "OpenSteer/Graphics/OpenGlRenderer.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Incude OpenSteer::Graphics::OpenGlRenderText
#include "OpenSteer/Graphics/OpenGlRenderText.h"

// Inlcude OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Graphics::createLine
#include "OpenSteer/Graphics/OpenGlUtilities.h"


int width = 640;
int height = 480;


OpenSteer::Vec3 position( 0.0f, 0.0f, 0.0f );

OpenSteer::Vec3 const step_left( -1.0f, 0.0f, 0.0f );
OpenSteer::Vec3 const step_up( 0.0f, 1.0f, 0.0f );
OpenSteer::Vec3 const step_forward( 0.0f, 0.0f, 1.0f );


OpenSteer::Graphics::OpenGlRenderer renderer;



void init()
{
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    glShadeModel( GL_FLAT );
}



void display() 
{
    glClear( GL_COLOR_BUFFER_BIT );
    
    GLfloat rasterPosition[4] = { -666.0f, -666.0f, -666.0f, -666.0f };
    glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
    std::cout << "0: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
    
    glLoadIdentity();
    
    glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
    std::cout << "1: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
    
    /*
    float transformation[ 16 ] = {  1.0f, 0.0f, 0.0f, 0.0f, // first column
                                    0.0f, 1.0f, 0.0f, 0.0f, // second column
                                    0.0f, 0.0f, 1.0f, 0.0f, // third column 
                                    0.0f, 0.0f, 0.0f, 1.0f }; // fourth column
    */
    
    std::cout << "position ( " << position[ 0 ] << ", " << position[ 1 ] << ", " << position[ 2 ] << " )" << std::endl;
    
    OpenSteer::Matrix transformation = OpenSteer::translationMatrix( position );
    
    
    std::cout << transformation( 0, 0 ) << ", " << transformation( 0, 1 ) << ", " << transformation( 0, 2 ) << ", " << transformation( 0, 3 ) << std::endl;
    std::cout << transformation( 1, 0 ) << ", " << transformation( 1, 1 ) << ", " << transformation( 1, 2 ) << ", " << transformation( 1, 3 ) << std::endl;
    std::cout << transformation( 2, 0 ) << ", " << transformation( 2, 1 ) << ", " << transformation( 2, 2 ) << ", " << transformation( 2, 3 ) << std::endl;
    std::cout << transformation( 3, 0 ) << ", " << transformation( 3, 1 ) << ", " << transformation( 3, 2 ) << ", " << transformation( 3, 3 ) << std::endl;
    
    glMultMatrixf( transformation.data() );
    
    glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
    std::cout << "2: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
    
    glColor3f( 1.0f, 1.0f, 1.0f );
    glRasterPos2i( 0, 0 );
    
    glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
    std::cout << "3: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
    
    
    
    
    std::ostringstream stream;
    stream << "(" << rasterPosition[ 0 ] << ", " << rasterPosition[ 1 ] << ", " << rasterPosition[ 2 ] << ", " << rasterPosition[ 3 ] << ")";
    
    std::string positionString( stream.str() );
    
    for ( std::string::size_type charPosition = 0; charPosition < positionString.length(); ++charPosition ) {
        glutBitmapCharacter (GLUT_BITMAP_9_BY_15, positionString[ charPosition ] );
    }
        
    glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
    std::cout << "4: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
    
    
    glutSwapBuffers();
}


void renderer_display() 
{
    glClear( GL_COLOR_BUFFER_BIT );
    
    OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderText > text( new OpenSteer::Graphics::OpenGlRenderText( std::string( "!\n*" ), OpenSteer::Color( 1.0f, 1.0f, 1.0f ) ) );
    OpenSteer::Matrix transformation = OpenSteer::translationMatrix( position );
    
    renderer.addToRender( transformation, OpenSteer::Graphics::createLine( OpenSteer::Vec3( -1.0f, 0.0f, 0.0f ), OpenSteer::Vec3( 1.0f, 0.0f, 0.0f ), OpenSteer::Color( 1.0f, 0.0f, 0.0f ) ) );
    renderer.addTextToRender( transformation, text );
    
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    gluLookAt( 0.0f, 0.0f, 5.0f,
               0.0f, 0.0f, -100.0f,
               0.0, 1.0f, 0.0f );
    renderer.render( OpenSteer::identityMatrix, OpenSteer::identityMatrix );
    renderer.renderText( OpenSteer::identityMatrix, OpenSteer::identityMatrix, width, height );
    renderer.precedeFrame();
    
    
    
    glutSwapBuffers();
}




void reshape( int w, int h )
{
    width = w;
    height = h;
    
    glViewport( 0, 0, static_cast< GLsizei >( w ), static_cast< GLsizei >( h ) );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    // glOrtho( -500.0f, 500.0f, -500.0f, 500.0f, -100.0f, 100.0f );
    gluPerspective( 80.0f, static_cast< GLfloat >( w ) / static_cast< GLfloat >( h ), 0.0f, 100000000.0f ); 
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
}


void mouse( int button, int state, int x, int y )
{
    
}


void specialkeys(int key, int x, int y)
{
    std::cout << "specialkeys" << std::endl;
    
    switch( key ) {
        case GLUT_KEY_LEFT:
            position += step_left;
            break;
        case GLUT_KEY_RIGHT:
            position -= step_left;
            break;
        case GLUT_KEY_UP:
            position += step_up;
            break;
        case GLUT_KEY_DOWN:
            position -= step_up;
            break;
        case GLUT_KEY_PAGE_UP:
            position += step_forward;
            break;
        case GLUT_KEY_PAGE_DOWN:
            position -= step_forward;
            break;
        case GLUT_KEY_HOME:
            position.set( 0.0f, 0.0f, 0.0f );
            break;
        default:
            // Nothing to do.
            break;
    } // switch
    
    glutPostRedisplay();
    
}


int main( int argc, char* argv[] )
{
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( argv[ 0 ] );
    init();
    // glutDisplayFunc( display );
    glutDisplayFunc( renderer_display );
    glutReshapeFunc( reshape );
    glutMouseFunc( mouse );
    glutSpecialFunc( specialkeys );
    
    glutMainLoop();
    
    return EXIT_SUCCESS;
}

