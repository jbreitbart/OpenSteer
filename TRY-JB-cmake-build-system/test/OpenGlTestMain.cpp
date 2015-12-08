/**
   @file
 
    Test for the OpenGL rendering components.
 */

// Include EXIT_SUCCESS
#include <cstdlib>

// Include std::cerr, std::cout, std::endl
#include <iostream>



// Include OpenGL, glu, glut.
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"

// Include OpenSteer:modulo
#include "OpenSteer/Utilities.h"

// Include OpenSteer::Graphics::OpenGlRenderPrimitive, OpenSteer::Graphics::createLine
#include "OpenSteer/Graphics/OpenGlRenderMesh.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Include OpenSteer::Graphics::OpenGlRenderer
#include "OpenSteer/Graphics/OpenGlRenderer.h"

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Graphics::createLine, OpenSteer::Graphics::createCircle, OpenSteer::Graphics::createFloor
#include "OpenSteer/Graphics/OpenGlUtilities.h"

// Include Opensteer::Graphics::OpenGlImage
#include "OpenSteer/Graphics/OpenGlImage.h"

// Include OpenSteer::Graphics::OpenGlTexture
#include "OpenSteer/Graphics/OpenGlTexture.h"


namespace {
    unsigned int meshId = 0;
    
    GLfloat spin = 0.0f;

    OpenSteer::Graphics::OpenGlRenderer renderer;
    
    
    void testForGlErrors( char const* msg ) {
    
        GLenum errorCode = glGetError();
        
        if ( GL_NO_ERROR != errorCode ) {
            GLubyte const* errorDescription = gluErrorString( errorCode );
            std::cerr << msg << " " << errorDescription << std::endl;
        }
        
    }
    
    
    void init() {
        glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
        glShadeModel( GL_FLAT );
    }

    
    void display() {
        using namespace OpenSteer;
        using namespace OpenSteer::Graphics;
        
        glClear( GL_COLOR_BUFFER_BIT );
        glPushMatrix();
        glRotatef( spin, 1.0f, 0.0f, 0.0f );
        
        
        SharedPointer< OpenGlRenderMesh> primitive( createLine( OpenSteer::Vec3( -5.0f, 0.0f, 0.0f ), 
																OpenSteer::Vec3( 5.0f, 0.0f, 0.0f ), 
																OpenSteer::Color( 0.0f, 0.0f, 1.0f ) ) );
        
        
        renderer.addToRender( primitive );
        
        // Replaced by a triangle drawn by its mesh id.
        // primitive.reset( new OpenGlRenderMesh( createVehicleTriangle( 5.0f, OpenSteer::Color( 1.0f, 0.0f, 0.0f ) ) ) );
        // renderer.addToRender( primitive );
        
        
        
        //primitive.reset( new OpenGlRenderMesh( createCircle( 20.0f, 20, OpenSteer::Color( 1.0f, 0.0f, 0.0f ) ) ) );
        // renderer.addToRender( translationMatrix( Vec3( 5.0f, 0.0f, 0.0f ) ), primitive );
        
        
        SharedPointer< OpenGlImage > floorImage( makeCheckerboardImage( 64, 64, 8, gWhite, gRed ) );
        
        // std::cout << "Image : " << *floorImage << std::endl;
        
        SharedPointer< OpenGlTexture > texture( new OpenGlTexture( floorImage ) );
        texture->setWrapS( OpenGlTexture::REPEAT );
        texture->setWrapT( OpenGlTexture::REPEAT );
        OpenGlTexture& txture = *texture; // For introspection in the debugger.
        OpenGlRenderer::TextureId textureId = 0;
        renderer.addTexture( texture, textureId );
        primitive = createFloor( 10.0f, 10.0f, gYellow, textureId );
        OpenGlRenderMesh& mesh = *primitive; // For introspection in the debugger.
        // print( mesh );
        renderer.addToRender( primitive );
        
        
        
        
        if ( false ) {
            glEnableClientState( GL_COLOR_ARRAY );
            glEnableClientState( GL_NORMAL_ARRAY );
            glEnableClientState( GL_VERTEX_ARRAY );
            
            float vertices[ 12 ] = { -5.0f, 0.0f, -5.0f,
                                    -5.0f, 0.0f, 5.0f,
                                    5.0f, 0.0f, 5.0f,
                                    5.0f, 0.0f, -5.0f };
            
            
            float colors[ 16 ] = { 0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f };
            float normals[ 12 ] = { 0.0f, 1.0f, 0.0f, 
                                    0.0f, 1.0f, 0.0f, 
                                    0.0f, 1.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f };
            unsigned int indices[ 4 ] = { 0, 1, 2, 3 };
            
            
            
            
            glColorPointer( 4, GL_FLOAT, 0, colors );
            glNormalPointer( GL_FLOAT, 0, normals );
            glVertexPointer( 3, GL_FLOAT, 0, vertices );
            
            glDrawElements( GL_LINE_LOOP, 4, GL_UNSIGNED_INT, indices ); 
        
            glDisableClientState( GL_VERTEX_ARRAY );
            glDisableClientState( GL_NORMAL_ARRAY );
            glDisableClientState( GL_COLOR_ARRAY );
            
        }
        
        if ( true ) {
    
            /*
            GLfloat const vertices[ 12 ] = { 
                -20.0f, -10.0f, 0.0f,
                -20.0f,  10.0f, 0.0f,
                  0.0f,  10.0f, 0.0f,
                  0.0f, -10.0f, 0.0f
                };
            
            GLfloat const colors[ 16 ] = { 
                0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f 
                };
            
            GLfloat const texCoords[ 8 ] = {
                0.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 1.0f,
                1.0f, 0.0f
                };
            
            GLuint const indices[ 4 ] = { 0, 1, 2, 3 };
            */
            
            
            OpenGlRenderMesh mesh( *(createFloor( 20.0f, 20.0f, gYellow, textureId ) ) );
            
            
            glEnableClientState( GL_TEXTURE_COORD_ARRAY );
            glEnableClientState( GL_COLOR_ARRAY );
            glEnableClientState( GL_NORMAL_ARRAY );
            glEnableClientState( GL_VERTEX_ARRAY );
            
            glEnable( GL_TEXTURE_2D );
            // glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glBindTexture( GL_TEXTURE_2D, mesh.textureId_ );
            glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, mesh.textureFunction_ );
            
            /*
            glTexCoordPointer( 2, GL_FLOAT, 0, texCoords );
            glColorPointer( 4, GL_FLOAT, 0, colors );
            glVertexPointer( 3, GL_FLOAT, 0, vertices );
            glDrawElements( GL_QUADS, 4, GL_UNSIGNED_INT, indices );
            */
            /*
            glTexCoordPointer( mesh.textureCoordinateElementCount_, GL_FLOAT, mesh.textureCoordinateStride_, &mesh.textureCoordinates_[ 0 ] );
            glColorPointer( 4, GL_FLOAT, 0, &mesh.colors_[ 0 ] );
            glVertexPointer( 3, GL_FLOAT, 0, &mesh.vertices_[ 0 ] );
            glDrawElements( mesh.type_, 4, GL_UNSIGNED_INT, &mesh.indices_[ 0 ] );
            */
            
            
            glColorPointer( mesh.colorElementCount_, GL_FLOAT, mesh.colorStride_, &mesh.colors_[ 0 ] );
            glNormalPointer( GL_FLOAT, mesh.normalStride_, &mesh.normals_[ 0 ] );
            glTexCoordPointer( mesh.textureCoordinateElementCount_, GL_FLOAT, mesh.textureCoordinateStride_, &mesh.textureCoordinates_[ 0 ]  );
            glVertexPointer( mesh.vertexElementCount_, GL_FLOAT, mesh.vertexStride_, &mesh.vertices_[ 0 ] );
            
            
            glDrawElements( mesh.type_, mesh.indices_.size(), GL_UNSIGNED_INT, &mesh.indices_[ 0 ] ); 
            
            
            glDisable( GL_TEXTURE_2D );
            glBindTexture( GL_TEXTURE_2D, 0 );
            
            glDisableClientState( GL_VERTEX_ARRAY );
            glDisableClientState( GL_NORMAL_ARRAY );
            glDisableClientState( GL_COLOR_ARRAY );
            glDisableClientState( GL_TEXTURE_COORD_ARRAY );
            
            
        }
        
        renderer.addToRender( meshId );
        
        renderer.render( OpenSteer::Matrix(), OpenSteer::Matrix() );
        
        renderer.removeTexture( textureId );
        
        
        testForGlErrors( "Any errors? " );
        
        renderer.precedeFrame();
        
        glPopMatrix();
        glutSwapBuffers();
    }

    void spinDisplay() {
        spin += 2.0f;
        spin = OpenSteer::modulo( spin, 360.0f );
        glutPostRedisplay();
    }
    
    void reshape( int w, int h ) {
        glViewport( 0, 0, static_cast< GLsizei>( w ), static_cast< GLsizei >( h ) );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho( -50.0f, 50.0f, -50.0f, 50.0f, -1.0f, 1.0f );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
    }
    
    void mouse( int button, int state, int x, int y ) {
        switch ( button ) {
            case GLUT_LEFT_BUTTON:
                if ( GLUT_DOWN == state ) {
                    glutIdleFunc( spinDisplay );
                }
                break;
            case GLUT_RIGHT_BUTTON:
                if ( GLUT_DOWN == state ) {
                    glutIdleFunc( 0 );
                }
                break;
            default:
                break;
        }
    }
    
} // anonymous namespace


int main( int argc, char** argv ) {
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
    glutInitWindowSize( 250, 250 );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( argv[ 0 ] );
    init();
    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutMouseFunc( mouse );
    
    // OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh> primitive( OpenSteer::Graphics::createVehicleTriangle( 5.0f, OpenSteer::Color( 0.0f, 1.0f, 0.0f ) ) );
    
    // OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh> primitive( new OpenSteer::Graphics::OpenGlRenderMesh( OpenSteer::Graphics::createLine( OpenSteer::Vec3( -3.0f, -3.0f, 0.0f ), OpenSteer::Vec3( 3.0f, 3.0f, 0.0f ), OpenSteer::Color( 0.0f, 1.0f, 0.0f ) ) ) );
    
	OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh> primitive( OpenSteer::Graphics::createBasic3dSphericalVehicle( 5.0f * 0.433f,
																	 5.0f * 0.933f,
																	 5.0f * 1.0f,
																	 5.0f * 0.5f, 
																	 OpenSteer::Color( 0.0f, 1.0f, 0.0f ),
																	 0.05f ) );
	
	
    renderer.addToRenderMeshLibrary( primitive, meshId );
    glutMainLoop();
    
    return EXIT_SUCCESS;
}


