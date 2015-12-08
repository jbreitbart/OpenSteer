/**
 * @todo Unify everything to use textures - the default texture seems to be ok
 *       if no valid texture is given.
 */

#include "OpenSteer/Graphics/OpenGlRenderer.h"


// Include std::cerr, std::endl
#include <iostream>

// Include std::transform
// #include <algorithm>

// Include std::advance
// #include <iterator>

// Include std::unary_function
#include <functional>

// Include std::vector
#include <vector>

// Include std::map
#include <map>

// Incude std::pair, std::make_pair
#include <utility>



// Include OpenGl, glut, glu
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"


// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::operator<<( ostr, Matrix const& )
#include "OpenSteer/MatrixUtilities.h"

// Include OpenSteer::Graphics::OpenGlRenderPrimitive
#include "OpenSteer/Graphics/OpenGlRenderMesh.h"

// Include OpenSteer::Graphics::OpenGlRenderText, OpenSteer::Graphics::OpenGlRenderText2d
#include "OpenSteer/Graphics/OpenGlRenderText.h"


// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

// Include KAPAGA_UNUSED_PARAMETER
#include "kapaga/unused_parameter.h"

namespace {
    

    using OpenSteer::Matrix;
    using OpenSteer::SharedPointer;
    using OpenSteer::Graphics::OpenGlRenderMesh;
    
    // Typedefs specifying the render queue containers for textured and 
    // non textured transformed meshes.
    // typedef std::pair< Matrix, GLuint > TransformedLibraryMesh;
    typedef std::pair< Matrix, SharedPointer< OpenGlRenderMesh > > TransformedMesh;
    typedef std::vector< TransformedMesh > TransformedMeshes;
    typedef GLuint TextureId;
    
    /**
     * Pairs the id of a texture and the associated dimensionality.
     */
    struct OpenGlTextureIdAndDimensionality {
        
        OpenGlTextureIdAndDimensionality() : id_( 0 ), dimensionality_( GL_TEXTURE_2D ) {
            // Nothing to do.
        }
        
        OpenGlTextureIdAndDimensionality( TextureId const& _id, GLenum const& _dimensionality ) : id_( _id ), dimensionality_( _dimensionality ) {
            // Nothing to do.
        }
        
        TextureId id_;
        GLenum dimensionality_;
    }; // struct OpenGlTextureIdAndDimensionality
    
    
    inline bool operator<( OpenGlTextureIdAndDimensionality const& lhs, OpenGlTextureIdAndDimensionality const& rhs ) {
        // @todo Is this ok?
        return ( lhs.id_ < rhs.id_ ) || ( lhs.dimensionality_ < rhs.dimensionality_ );
    }
    
    
    typedef std::map< OpenGlTextureIdAndDimensionality, TransformedMeshes > TexturedTransformedMeshes;
    typedef TexturedTransformedMeshes::value_type TextureIdAndTransformedMeshes;
    
    
    
    struct PrepareUntexturedUntransformedMeshForDrawing : public std::unary_function< OpenGlRenderMesh, void > {
      
        void operator()( OpenGlRenderMesh const& _mesh ) const {
            glColorPointer( _mesh.colorElementCount_, GL_FLOAT, _mesh.colorStride_, &_mesh.colors_[ 0 ] );
            glNormalPointer( GL_FLOAT, _mesh.normalStride_, &_mesh.normals_[ 0 ] );
            glVertexPointer( _mesh.vertexElementCount_, GL_FLOAT, _mesh.vertexStride_, &_mesh.vertices_[ 0 ] );
        }
        
    }; // struct PrepareUntexturedUntransformedMeshForDrawing
    
    
    struct DrawUntransformedMesh : public std::unary_function< OpenGlRenderMesh, void > {
        void operator()( OpenGlRenderMesh const& _mesh ) const {
			// Ugly hack to add wireframe mode - this should be sorted, too. 
			glPolygonMode( GL_FRONT, _mesh.polygonMode_ );
            glDrawElements( _mesh.type_, _mesh.indices_.size(), GL_UNSIGNED_INT, &_mesh.indices_[ 0 ] );
        }
    }; // struct DrawUntexturedUntransformedMesh
    
    
    
    
    
    /**
     * Functor to render @a _mesh without setting up the render environment 
     * before.
     *
     * Put this into a scope with @c UntexturedVertexArrayRenderingConfig or
     * @c TexturedVertexArrayRenderingConfig.
     */
    struct RenderUntexturedUntransformedMesh : public std::unary_function< OpenGlRenderMesh, void > {
        void operator()( OpenGlRenderMesh const& _mesh ) const {
            PrepareUntexturedUntransformedMeshForDrawing()( _mesh );
            DrawUntransformedMesh()( _mesh );               
        }
    }; // RenderUntexturedUntransformedMesh
    
    
    /**
     * Functor to render a transformed mesh residing in the renderer library
     * without setting up the render environment before.
     *
     * Put this into a scope with @c UntexturedVertexArrayRenderingConfig or
     * @c TexturedVertexArrayRenderingConfig.
     */
    /*
    struct RenderLibraryMesh {
        
        void operator()( TransformedLibraryMesh& _libraryMesh ) {
            // @todo Add support for the projection and modelview matrix and remove glMatrixPush/pop.
            glPushMatrix();
            glMultMatrixf( _libraryMesh.first.data() );
            glCallList( _libraryMesh.second );
            glPopMatrix();
        }
        
    };
    */
    
    /**
     * Functor to render a transformed mesh without setting up the render 
     * environment before.
     *
     * Put this into a scope with @c UntexturedVertexArrayRenderingConfig or
     * @c TexturedVertexArrayRenderingConfig.
     */
    struct RenderUntexturedTransformedMesh : std::unary_function< TransformedMesh, void > {
        void operator() ( TransformedMesh const& _mesh ) const {
            
            // @todo Add support for the projection and modelview matrix and remove glMatrixPush/pop.
            glPushMatrix();
            glMultMatrixf( _mesh.first.data() );
            RenderUntexturedUntransformedMesh()( *(_mesh.second) );
            glPopMatrix();
        }
    }; // struct RenderUntexturedTransformedMesh
    
    
    /**
     * Uses the resource-aquisition-is-initialization (RAII) idiom
     * to configure and unconfigure OpenGL to render untextured vertex arrays.
     */
    class UntexturedVertexArrayRenderingConfig {
    public:
        UntexturedVertexArrayRenderingConfig() {
            glEnableClientState( GL_COLOR_ARRAY );
            glEnableClientState( GL_NORMAL_ARRAY );
            glEnableClientState( GL_VERTEX_ARRAY );
        }
        
        ~UntexturedVertexArrayRenderingConfig() {
            glDisableClientState( GL_VERTEX_ARRAY );
            glDisableClientState( GL_NORMAL_ARRAY );
            glDisableClientState( GL_COLOR_ARRAY );
        }
        
    }; // UntexturedVertexArrayRenderingConfig
    
    
    /**
     * Uses the resource-aquisition-is-initialization (RAII) idiom
     * to configure and unconfigure OpenGL to render textured
     * vertex arrays.
     */
    class TexturedVertexArrayRenderingConfig {
    public:
        TexturedVertexArrayRenderingConfig() {
            glEnableClientState( GL_COLOR_ARRAY );
            glEnableClientState( GL_NORMAL_ARRAY );
            glEnableClientState( GL_TEXTURE_COORD_ARRAY );
            glEnableClientState( GL_VERTEX_ARRAY );
        }
        
        ~TexturedVertexArrayRenderingConfig() {
            glDisableClientState( GL_VERTEX_ARRAY );
            glDisableClientState( GL_TEXTURE_COORD_ARRAY );
            glDisableClientState( GL_NORMAL_ARRAY );
            glDisableClientState( GL_COLOR_ARRAY );
        }
        
    }; // TexturedVertexArrayRenderingConfig
    
    
    
    /**
     * @attention For the RAII (resource aquisition is initializatio) idiom to 
     *            work the instance must be named.
     */
    class OpenGlTextureObjectBind {
    public:        
        OpenGlTextureObjectBind( TextureId const& _id, GLenum _dimensionality ) : data_( _id, _dimensionality )  {
            glBindTexture( data_.dimensionality_, data_.id_ );
        }
        
        OpenGlTextureObjectBind( OpenGlTextureIdAndDimensionality const& _info ) : data_( _info ) {
            glBindTexture( data_.dimensionality_, data_.id_ );
        }
        
        ~OpenGlTextureObjectBind() {
            glBindTexture( data_.dimensionality_, defaultTexture_ );
        }
        
    private:
        /**
        * Intentionally not implemented to prevent copying.
         */
        OpenGlTextureObjectBind( OpenGlTextureObjectBind const& );
        
        /**
        * Intentionally not implemented to prevent copying.
         */
        OpenGlTextureObjectBind& operator=(  OpenGlTextureObjectBind const& );
        
    private:
            OpenGlTextureIdAndDimensionality data_;
        static TextureId const defaultTexture_ = 0;
        
    }; // class OpenGlTextureObjectBind
    
    
    
    
    /**
     * Representation of an OpenGL texture object to automate creation and 
     * destruction of texture objects.
     *
     * @todo Add functionality to handle texture object attribute changes.
     *
     * @todo Add a base class and inherit a class for 1d textures from it.
     */
    class OpenGlTexture2dObject {
    public:
        typedef GLuint TextureId;
        
        OpenGlTexture2dObject( OpenSteer::Graphics::OpenGlTexture const& _texture ) : id_( 0 ) {            
            glGenTextures( 1, &id_ );
            bool const setTextureSuccessful = setTexture( _texture );

            if ( ! setTextureSuccessful ) {
                glDeleteTextures( 1, &id_ );
                id_ = 0;
            }
            
        }
        
        
        ~OpenGlTexture2dObject() {
            glDeleteTextures( 1, &id_ );
        }
        
        bool setTexture( OpenSteer::Graphics::OpenGlTexture const& _texture ) {
            if (  ! textureAssignable( _texture ) ) {
                return false;
            }
            
            GLenum const dimensionality = _texture.image()->glDimensionality();
            OpenGlTextureObjectBind textureBinding( id_, dimensionality );
            
            glTexParameteri( dimensionality, GL_TEXTURE_WRAP_S, _texture.wrapS() );
            glTexParameteri( dimensionality, GL_TEXTURE_WRAP_T, _texture.wrapT() );
            glTexParameteri( dimensionality, GL_TEXTURE_MAG_FILTER, _texture.magnificationFilter() );
            glTexParameteri( dimensionality, GL_TEXTURE_MIN_FILTER, _texture.minificationFilter() );
            glTexEnvfv( GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, _texture.borderColor().colorFloatArray() );
            float const priority = _texture.priority();
            glPrioritizeTextures( 1, &id_, &priority );
            // @todo Handle this in a cleaner way, however currently only 
            //       one texture level is used, no mipmapping.
            GLint const textureLevel = 0;
            glPixelStorei( GL_UNPACK_ALIGNMENT, _texture.image()->glUnpackAlignment() );
            
            // @todo Handle textures with different dimensions.
            glTexImage2D( GL_TEXTURE_2D, 
                          textureLevel, 
                          GL_RGBA, 
                          _texture.image()->width(), 
                          _texture.image()->height(), 
                          _texture.border(), 
                          _texture.image()->glPixelFormat(), 
                          _texture.image()->glPixelType(), 
                          _texture.image()->data() ); 
            
            return true;
        }
        
        
        TextureId id() const {
            return id_;
        }
        
        bool valid() const {
            return glIsTexture( id_ );
        }
        
        GLenum dimensionality() const {
            return GL_TEXTURE_2D;
        }
        
        /**
         * Returns @c true if the texture contains an @c OpenGlImage and if the
         * dimensionality of the texture/its image is the same as 
         * @c dimensionality(), returns @c false otherwise.
         */ 
        bool textureAssignable( OpenSteer::Graphics::OpenGlTexture const& _texture ) const {
            return _texture.image() && ( dimensionality() == _texture.image()->glDimensionality() ) ;
        }
        
    private:
    
        /**
         * Intentionally not implemented to disable copying.
         */
        OpenGlTexture2dObject( OpenGlTexture2dObject const& );
        
        /**
         * Intentionally not implemented to disable copying.
         */
        OpenGlTexture2dObject& operator=( OpenGlTexture2dObject const& );
    
    private:
            
        TextureId id_;
    }; // class OpenGlTexture2dObject
    
    

    
    struct PrepareTexturedUntransformedMeshForDrawing :  public std::unary_function< OpenGlRenderMesh, void > {
        void operator()( OpenGlRenderMesh const& _mesh ) const {
            glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, _mesh.textureFunction_ );
            
            glColorPointer( _mesh.colorElementCount_, GL_FLOAT, _mesh.colorStride_, &_mesh.colors_[ 0 ] );
            glNormalPointer( GL_FLOAT, _mesh.normalStride_, &_mesh.normals_[ 0 ] );
            glTexCoordPointer( _mesh.textureCoordinateElementCount_, GL_FLOAT, _mesh.textureCoordinateStride_, &_mesh.textureCoordinates_[ 0 ]  );
            glVertexPointer( _mesh.vertexElementCount_, GL_FLOAT, _mesh.vertexStride_, &_mesh.vertices_[ 0 ] );
        }
    }; // struct PrepareTexturedUntransformedMeshForDrawing
    
    
/*
    struct DrawTexturedUntransformedMesh :  public std::unary_function< OpenGlRenderMesh, void > {
        void operator()( OpenGlRenderMesh const& _mesh ) {
            glDrawElements( _mesh.type_, _mesh.indices_.size(), GL_UNSIGNED_INT, &_mesh.indices_[ 0 ] );
        }
        
    }; // struct DrawTexturedUntransformedMesh
*/  
    
    /**
     * Functor to render @a _mesh without setting up the render environment 
     * before. @a _mesh is textured.
     *
     * Put this into a scope with @c UntexturedVertexArrayRenderingConfig or
     * @c TexturedVertexArrayRenderingConfig.
     */
    struct RenderTexturedUntransformedMesh : public std::unary_function< OpenGlRenderMesh, void > {
        void operator()( OpenGlRenderMesh const& _mesh ) const {
            
            PrepareTexturedUntransformedMeshForDrawing()( _mesh );
            DrawUntransformedMesh()( _mesh );
        }
    }; // RenderTexturedTransformedMesh
    
    
    /**
     * Functor to render a transformed textured mesh without setting up the  
     * render environment before.
     *
     * Put this into a scope with @c UntexturedVertexArrayRenderingConfig or
     * @c TexturedVertexArrayRenderingConfig.
     *
     * @todo Fusion it with its untextured brother into a function that takes a 
     *       functor argument how to render the mesh.
     */
    struct RenderTexturedTransformedMesh : std::unary_function< TransformedMesh, void > {
        void operator() ( TransformedMesh const& _mesh ) const {
            
            // @todo Add support for the projection and modelview matrix and remove glMatrixPush/pop.
            glPushMatrix();
            glMultMatrixf( _mesh.first.data() );
            RenderTexturedUntransformedMesh()( *(_mesh.second) );
            glPopMatrix();
        }
    }; // struct RenderTtexturedTransformedMesh
    
    
    
    
    class OpenGlEnableTexturing {
    public:    
        OpenGlEnableTexturing( GLenum _dimensionality ) : textureDimension_( _dimensionality ) {
            glEnable( textureDimension_ );
        }
        
        
        ~OpenGlEnableTexturing() {
            glDisable( textureDimension_ );
        }
        
    private:
        GLenum textureDimension_;
    }; // class OpenGlEnableTexturing
    
    
    
    
    

    class RenderTextureIdAndTransformedMeshes : public std::unary_function< TextureIdAndTransformedMeshes, void > {
    public:
        
        void operator() ( TextureIdAndTransformedMeshes const& _mesh ) const {
            OpenGlEnableTexturing textureEnabler( _mesh.first.dimensionality_ );
            OpenGlTextureObjectBind textureBinder( _mesh.first.id_, _mesh.first.dimensionality_ );
            std::for_each( _mesh.second.begin(), _mesh.second.end(), RenderTexturedTransformedMesh() );
        }

        
    }; // class RenderTexturedTransformedMeshes
    
    
    
//    class RenderTexturedInstances : public std::unary_function< InstanceIdAndTransformation, void > {
//    public:
//        
//        void operator()( InstanceIdAndTransformation const& 
//        
//    }; // class RenderTexturedInstances
    
    
    // @todo Unify the type declarations for the helper functors and the renderer.
    typedef OpenSteer::Graphics::OpenGlRenderer::InstanceId RenderMeshLibraryId;
    typedef std::map< RenderMeshLibraryId, OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh > > RenderMeshLibrary;
   
    typedef std::vector< OpenSteer::Matrix > Transformations;
    typedef std::map< RenderMeshLibraryId, Transformations > Instances;
    
    typedef std::map< TextureId, SharedPointer< OpenGlTexture2dObject >  > TextureIdLibrary;
    
    class RenderInstance : public std::unary_function< Transformations, void > {
    public:
        
        // Assumes that @a _id is contained in @a _idLibrary.
        RenderInstance( OpenSteer::Graphics::OpenGlRenderMesh const* _mesh )
            : mesh_( _mesh ) {
            // Nothing to do.
        }
        
        
        void operator() ( OpenSteer::Matrix const& _transformation ) const {
            
            glPushMatrix();
            glMultMatrixf( _transformation.data() );
            DrawUntransformedMesh()( *mesh_ );
            glPopMatrix();
        }
        
    private:
        
        OpenSteer::Graphics::OpenGlRenderMesh const* mesh_;
        
    }; // class RenderInstance
    
    
    
    class RenderUntexturedInstances : public std::unary_function< Instances::value_type, void > {
    public:
        
        RenderUntexturedInstances( RenderMeshLibrary const* _renderMeshLibrary ) : renderMeshLibrary_( _renderMeshLibrary ) {
            
        }
        
        void operator() ( Instances::value_type const& _instancesOfOneId ) const {
            
            RenderMeshLibrary::const_iterator meshIter = renderMeshLibrary_->find( _instancesOfOneId.first );
            
            assert( renderMeshLibrary_->end() != meshIter && "Instance to render isn't contained in renderMeshLibrary_." );
            
            OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh > const mesh = meshIter->second;
            
            PrepareUntexturedUntransformedMeshForDrawing()( *mesh );
            std::for_each( _instancesOfOneId.second.begin(), _instancesOfOneId.second.end(), RenderInstance( mesh.get() ) );
        }
        
    private:
        RenderMeshLibrary const* renderMeshLibrary_;
        
    }; // class RenderUntexturedInstances
    
    
    

    
    
    
    class RenderTexturedInstances : public std::unary_function< Instances::value_type, void > {
    public:
        
        RenderTexturedInstances( RenderMeshLibrary const* _renderMeshLibrary, TextureIdLibrary const* _textureIdLibrary ) : renderMeshLibrary_( _renderMeshLibrary ), textureIdLibrary_( _textureIdLibrary ) {
            
        }
        
        void operator() ( Instances::value_type const& _instancesOfOneId ) const {
            
            RenderMeshLibrary::const_iterator meshIter = renderMeshLibrary_->find( _instancesOfOneId.first );
            
            assert( renderMeshLibrary_->end() != meshIter && "Instance to render isn't contained in renderMeshLibrary_." );
            
            OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderMesh > const mesh = meshIter->second;
            
            RenderMeshLibraryId const textureId = mesh->textureId_;
            TextureIdLibrary::const_iterator textureIter = textureIdLibrary_->find( textureId );
            
            assert( ( textureIdLibrary_->end() != textureIter ) && "Texture not contained in textureIdLibrary_." );
            
            OpenSteer::SharedPointer< OpenGlTexture2dObject > const textureObject = textureIter->second;
            
            OpenGlEnableTexturing textureEnabler( textureObject->dimensionality() );
            OpenGlTextureObjectBind textureBinder( textureId, textureObject->dimensionality() );
            PrepareTexturedUntransformedMeshForDrawing()( *mesh );
            std::for_each( _instancesOfOneId.second.begin(), _instancesOfOneId.second.end(), RenderInstance( mesh.get() ) );
        }
        
    private:
        RenderMeshLibrary const* renderMeshLibrary_;
        TextureIdLibrary const* textureIdLibrary_;
        
    }; // class RenderUntexturedInstances
    
    
    
    typedef std::pair< Matrix, OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderText > > Text3d;
    // typedef std::vector< Text3d > Text3DRenderQueue;
    
    class RenderText3d : public std::unary_function< OpenSteer::SharedPointer< Text3d >, void > {
    public:
        
        void operator()( Text3d const& _text ) {
            
            
            glColor3f ( _text.second->material_.r(), _text.second->material_.g(), _text.second->material_.b());
            
            // GLint rasterPosition[4];
            //glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
            //std::cout << "RenderText3d::operator() 1: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
            
            // Set transformation matrix to influence raster positioning.
            glPushMatrix();
            glMultMatrixf( _text.first.data() );
            //std::cout << "Text transformation " << std::endl;
            //std::cout << _text.first << std::endl;
            
            //size_t const matrixElementCount = 16;
            //float modelViewMatrix[ matrixElementCount ];
            //glGetFloatv( GL_MODELVIEW_MATRIX, modelViewMatrix );
            //Matrix transform( &modelViewMatrix[ 0 ], &modelViewMatrix[ matrixElementCount ] );
            //std::cout << "Modelview Transformation" << std::endl;
            //std::cout << transform << std::endl;
            
            
            
            glRasterPos3f ( 0.0f, 0.0f, 0.0f );
            //glRasterPos3f( _text.first[ 12 ], _text.first[ 13 ], _text.first[ 14 ] );
            
            GLfloat rasterPosition[4];
            glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
            // std::cout << "RenderText3d::operator() 2: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << " " << _text.second->text_ << std::endl;
            
            
            
            
            
            // @todo Rewrite using find and for_each to render text.
            
            // loop over each character in string (until null terminator)
            int lines = 0;
            // @todo Remove hard coded font data from here and put it into the
            //       parameters of the text class.
            int const fontHeight = 15; // for GLUT_BITMAP_9_BY_15
            
            // std::cout << "text: " << _text.second->text_ << std::endl;
            
            glMatrixMode( GL_PROJECTION );
            glPushMatrix();
            glLoadIdentity();
            int const windowWidth = glutGet (GLUT_WINDOW_WIDTH);
            int const windowHeight = glutGet (GLUT_WINDOW_HEIGHT);
            glOrtho (0.0f, static_cast< float >( windowWidth ), 0.0f, static_cast< float >( windowHeight ), -1.0f, 1.0f); 
            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glLoadIdentity();
            
            for (const char* p = _text.second->text_.c_str(); *p; ++p )
            {
                if ( '\n' != *p ) {
                    #ifndef HAVE_NO_GLUT
                    
                    // glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    // glRasterPos3i( rasterPosition[ 0 ], rasterPosition[ 1 ], 1 );
                    // std::cout << "RenderText3d::operator() 3: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << " " << *p << std::endl;
                    
                    glutBitmapCharacter (GLUT_BITMAP_9_BY_15, *p);
                    
                    // glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    // std::cout << "RenderText3d::operator() 4: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] <<  " " << *p << std::endl;
                    
                    
                    #else
                    // no character drawing with GLUT presently
                    #endif
                    
                } else {
                    // std::cout << "Linebreak" << std::endl;
                    ++lines;                    
                    const int vOffset = lines * (fontHeight + 1);
                    //glRasterPos2i ( 0, -vOffset);  
                    //glRasterPos3f( _text.first[ 12 ], _text.first[ 13 ]-vOffset, _text.first[ 14 ] );
                    
                    //glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    //std::cout << "RenderText3d::operator() vor 5: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] <<  " " << *p << std::endl;
                    glRasterPos3f( rasterPosition[ 0 ], rasterPosition[ 1 ] - vOffset, rasterPosition[ 2 ] );
                    
                    //glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    //std::cout << "RenderText3d::operator() 5: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] <<  " " << *p << std::endl;
                    
                    
                } // if else
                
            } // for
            
            glMatrixMode( GL_PROJECTION );
            glPopMatrix();
            
            glMatrixMode( GL_MODELVIEW );
            glPopMatrix();
            glPopMatrix();
        }
        
        
    }; // class RenderText3d
    
    
    typedef OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlRenderText2d > Text2d;
    
    class RenderText2d : public std::unary_function< Text2d, void > {
    public:
        
        void operator()( Text2d const& _text ) {
            
            using OpenSteer::Graphics::OpenGlRenderText2d;
            
            glColor3f ( _text->material_.r(), _text->material_.g(), _text->material_.b() );
            
            
            // GLint rasterPosition[4];
            //glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
            //std::cout << "RenderText3d::operator() 1: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << std::endl;
            
            // Set transformation matrix to influence raster positioning.
            // glPushMatrix();
            // glMultMatrixf( _text.first.data() );
            //std::cout << "Text transformation " << std::endl;
            //std::cout << _text.first << std::endl;
            
            //size_t const matrixElementCount = 16;
            //float modelViewMatrix[ matrixElementCount ];
            //glGetFloatv( GL_MODELVIEW_MATRIX, modelViewMatrix );
            //Matrix transform( &modelViewMatrix[ 0 ], &modelViewMatrix[ matrixElementCount ] );
            //std::cout << "Modelview Transformation" << std::endl;
            //std::cout << transform << std::endl;
            
            float const windowWidth = static_cast< float >( glutGet( GLUT_WINDOW_WIDTH ) );
            float const windowHeight = static_cast< float >( glutGet( GLUT_WINDOW_HEIGHT ) );
            OpenSteer::Vec3 position( _text->position_ );
            
            // The origin of the screen coordinate system is at the bottom left.
            switch ( _text->relativePosition_ ) {
                case OpenGlRenderText2d::TOP_LEFT:
                    position[ 1 ] = windowHeight - position[ 1 ];
                    break;
                case OpenGlRenderText2d::BOTTOM_LEFT:
                    // Nothing to do.
                    break;
                case OpenGlRenderText2d::TOP_RIGHT:
                    position[ 1 ] = windowHeight - position[ 1 ];
                    position[ 0 ] = windowWidth - position[ 0 ];
                    break;
                case OpenGlRenderText2d::BOTTOM_RIGHT:
                    position[ 0 ] = windowWidth - position[ 0 ];
                    break;
                default:
                    std::cerr << "RenderText2d::operator() : Unknown relative position of text." << std::endl;
                    break;
            }
            
            
            
            glRasterPos3f ( position[ 0 ], position[ 1 ], position[ 2 ] );
            //glRasterPos3f( _text.first[ 12 ], _text.first[ 13 ], _text.first[ 14 ] );
            
            GLfloat rasterPosition[4];
            glGetFloatv (GL_CURRENT_RASTER_POSITION, rasterPosition);
            // std::cout << "RenderText3d::operator() 2: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << " " << _text.second->text_ << std::endl;
            
            
            
           //  std::cout << "Drawing Color : " << _text->material_.r() << " " << _text->material_.g() << " " << _text->material_.b() << " Text: " <<  _text->text_ << std::endl;
            
            
            
            
            // @todo Rewrite using find and for_each to render text.
            
            // loop over each character in string (until null terminator)
            int lines = 0;
            // @todo Remove hard coded font data from here and put it into the
            //       parameters of the text class.
            int const fontHeight = 15; // for GLUT_BITMAP_9_BY_15
            
            // std::cout << "text: " << _text.second->text_ << std::endl;
            
            glMatrixMode( GL_PROJECTION );
            glPushMatrix();
            glLoadIdentity();
            
            glOrtho (0.0f, static_cast< float >( windowWidth ), 0.0f, static_cast< float >( windowHeight ), -1.0f, 1.0f); 
            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glLoadIdentity();
            
            for (const char* p = _text->text_.c_str(); *p; ++p )
            {
                if ( '\n' != *p ) {
#ifndef HAVE_NO_GLUT
                    
                    // glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    // glRasterPos3i( rasterPosition[ 0 ], rasterPosition[ 1 ], 1 );
                    // std::cout << "RenderText3d::operator() 3: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] << " " << *p << std::endl;
                    
                    glutBitmapCharacter (GLUT_BITMAP_9_BY_15, *p);
                    
                    // glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    // std::cout << "RenderText3d::operator() 4: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] <<  " " << *p << std::endl;
                    
                    
#else
                    // no character drawing with GLUT presently
#endif
                    
                } else {
                    // std::cout << "Linebreak" << std::endl;
                    ++lines;                    
                    const int vOffset = lines * (fontHeight + 1);
                    //glRasterPos2i ( 0, -vOffset);  
                    //glRasterPos3f( _text.first[ 12 ], _text.first[ 13 ]-vOffset, _text.first[ 14 ] );
                    
                    //glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    //std::cout << "RenderText3d::operator() vor 5: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] <<  " " << *p << std::endl;
                    glRasterPos3f( rasterPosition[ 0 ], rasterPosition[ 1 ] - vOffset, rasterPosition[ 2 ] );
                    
                    //glGetIntegerv (GL_CURRENT_RASTER_POSITION, rasterPosition);
                    //std::cout << "RenderText3d::operator() 5: " << rasterPosition[ 0 ] << " " << rasterPosition[ 1 ] << " " << rasterPosition[ 2 ] << " " << rasterPosition[ 3 ] <<  " " << *p << std::endl;
                    
                    
                } // if else
                
            } // for
            
            glMatrixMode( GL_PROJECTION );
            glPopMatrix();
            
            glMatrixMode( GL_MODELVIEW );
            glPopMatrix();
        }
        
        
    }; // class RenderText2d
    
    
    
    
    
} // anonymous namespace




namespace OpenSteer {
    
    
    namespace Graphics {
        
        /**
         * Helper structure holding different render qeues for different 
         * OpenGl render meshes or meshes with different attributes.
         *
         * Renders the meshes using vertex arrays. Even the meshes stored in the
         * render mesh library are finally rendered (if their associated id is
         * added to render it) using vertex arrays.
         *
         * Simplistic meshes like lines, triangles, points, or quads without 
         * transformations and textures are merged into just one mesh.
         *
         * To render the meshes added one of the @c render functions has to be
         * called, for example @c renderAll.
         */
        struct OpenGlRenderer::Impl {
            
            Impl()
            : lines_( OpenGlRenderMesh::LINES ), 
            triangles_( OpenGlRenderMesh::TRIANGLES ),
            points_( OpenGlRenderMesh::POINTS ),
            quads_( OpenGlRenderMesh::QUADS ),
            /* libraryRenderQueue_(), */
            untexturedRenderQueue_(),
            texturedRenderQueue_(),
            renderMeshLibrary_(),
            nextRenderMeshLibraryId_( 1 ),
            textureIdLibrary_(),
            untexturedInstanceRenderQueue_(),
            texturedInstanceRenderQueue_(),
            text3dRenderQueue_(),
            text2dRenderQueue_() {
                // Nothing to do.
            }
            
            ~Impl() {
                // @todo Free the display lists.
            }
            
            /**
             * Returns @c true if the mesh was successfully added to the render
             * mesh library and @a _id 
             * will be a value greater than @c 0. Otherwise @c false is returned
             * and @a _id will be set to @c 0.
             */
            bool addToRenderMeshLibrary( SharedPointer< OpenGlRenderMesh > const& _mesh, 
                                         OpenGlRenderer::InstanceId& _id ) {
                
                if ( renderMeshLibrary_.end() != renderMeshLibrary_.find( nextRenderMeshLibraryId_ ) ) {
                    // The next id is already in the library. Too many meshes are stored in the library.
                    return false;
                }
                                
                renderMeshLibrary_[ nextRenderMeshLibraryId_ ] = _mesh;
                _id = nextRenderMeshLibraryId_++;
                return true;
            }
            
            void removeFromRenderMeshLibrary( OpenGlRenderer::InstanceId const& _id ) {
                renderMeshLibrary_.erase( _id );
            }
            
            bool inRenderMeshLibrary( OpenGlRenderer::InstanceId const& _id ) {
                return renderMeshLibrary_.end() != renderMeshLibrary_.find( _id );
            }
            
            void clearRenderMeshLibrary() {
                renderMeshLibrary_.clear();
            }
            
            
            void clearRenderQueues() {
                lines_.clear();
                triangles_.clear();
                points_.clear();
                quads_.clear();
                untexturedRenderQueue_.clear();
                texturedRenderQueue_.clear();
                untexturedInstanceRenderQueue_.clear();
                texturedInstanceRenderQueue_.clear();
                text3dRenderQueue_.clear();
                text2dRenderQueue_.clear();
                
            }
            
            /**
             * Put @a _mesh into the appropriate render queue.
             */
            void add( SharedPointer< OpenGlRenderMesh > const& _mesh ) {
                // @todo bknafla Test to unify the rendering pipeline paths - slows down the rendering!
                if ( !OpenSteer::Graphics::textured( *_mesh ) ) {
                    switch ( _mesh->type_ ) {
                        case OpenGlRenderMesh::LINES:
                            OpenSteer::Graphics::mergeMeshesFast( lines_, *_mesh );
                            break;
                        case OpenGlRenderMesh::TRIANGLES:
                            OpenSteer::Graphics::mergeMeshesFast( triangles_, *_mesh );
                            break;
                        case OpenGlRenderMesh::POINTS:
                            OpenSteer::Graphics::mergeMeshesFast( points_, *_mesh );
                            break;
                        case OpenGlRenderMesh::QUADS:
                            OpenSteer::Graphics::mergeMeshesFast( quads_, *_mesh );
                            break;
                        default:
                            // OpenGL geometry types other than above can't be 
                            // merged and don't get a special treatment though
                            // they are untransformed.
                            add( OpenSteer::identityMatrix, _mesh );
                            break;
                    }
                } else  {
                    // Textured meshes can't be merged and don't get a special
                    // treatment though they are untransformed.
                    add( OpenSteer::identityMatrix, _mesh );
                };
                
            }
            
            /**
             * Put the transformed @a _mesh into the appropriate render queue.
             */
            void add( Matrix const& _transformation, SharedPointer< OpenGlRenderMesh > const& _mesh ) {
                if ( OpenSteer::Graphics::textured(* _mesh ) ) {
                    assert( containsTexture( _mesh->textureId_ ) && "_mesh texture not contained in renderer." );
                    
                    ( texturedRenderQueue_[ OpenGlTextureIdAndDimensionality( _mesh->textureId_, textureIdLibrary_[ _mesh->textureId_ ]->dimensionality() ) ] ).push_back( std::make_pair( _transformation, _mesh ) );
                } else {
                    // @todo bknafla: Test what happens if I draw everything as textured.
                    untexturedRenderQueue_.push_back( std::make_pair( _transformation, _mesh ) );
                    //( texturedRenderQueue_[ OpenGlTextureIdAndDimensionality( _mesh->textureId_, textureIdLibrary_[ _mesh->textureId_ ]->dimensionality() ) ] ).push_back( std::make_pair( _transformation, _mesh ) );
                }
                
            }
            
            // @todo Collect transformations for the same ids and the draw all the same ids by enabling their arrays just once and setting their transformation and then call draw for the arrays - perhaps lesser movement of data to OpenGL hardware needed?
            bool add( Matrix const& _transformation, 
                      OpenGlRenderer::InstanceId const& _id ) {
            
                if ( !inRenderMeshLibrary( _id ) ) {
                    return false;
                }
                
                // libraryRenderQueue_.push_back( std::make_pair( _transformation, _id ) );
                
                // @todo bknafla: Reenable optimized instances rendering
                // add( _transformation, renderMeshLibrary_[ _id ] );
                // return true;
                
                if ( OpenSteer::Graphics::textured( *renderMeshLibrary_[ _id ] ) ) {
                    // Textured instance.
                    texturedInstanceRenderQueue_[ _id ].push_back( _transformation );
                } else {
                    // Untextured instance.
                    // @todo bknafla: handle everything as if it is textured
                    untexturedInstanceRenderQueue_[ _id ].push_back( _transformation );
                }
                
                return true;
            }
            
            
            
            void addText( Matrix const& _transformation,
                          SharedPointer< OpenGlRenderText > const& _text ) {
                text3dRenderQueue_.push_back( std::make_pair( _transformation, _text ) );                
            }
            
            void addText2d( Text2d const& _text  ) {
                text2dRenderQueue_.push_back( _text );
            }
            
            
            void renderAllUntexturedUntransformedMeshes() {
                {
                    UntexturedVertexArrayRenderingConfig config;
                    
                    RenderUntexturedUntransformedMesh()( lines_ );
                    RenderUntexturedUntransformedMesh()( triangles_ );
                    RenderUntexturedUntransformedMesh()( points_ );
                    RenderUntexturedUntransformedMesh()( quads_ );
                }
                lines_.clear();
                triangles_.clear();
                points_.clear();
                quads_.clear();
            }
            
            
            /*
            void renderLibraryRenderQueue() {
                std::for_each( libraryRenderQueue_.begin(), libraryRenderQueue_.end(), RenderLibraryMesh()  );
                libraryRenderQueue_.clear();
            }
            */
            
            void renderUntexturedRenderQueue() {
                {
                    UntexturedVertexArrayRenderingConfig config;
                    std::for_each( untexturedRenderQueue_.begin(), untexturedRenderQueue_.end(), RenderUntexturedTransformedMesh() );
                }
                untexturedRenderQueue_.clear();
            }
            
            
            
            void renderTexturedRenderQueue() {
                
                {
                    TexturedVertexArrayRenderingConfig config;
                    std::for_each( texturedRenderQueue_.begin(), texturedRenderQueue_.end(), RenderTextureIdAndTransformedMeshes() );
                }
                texturedRenderQueue_.clear();
            }
            
            
            
            void renderUntexturedInstanceRenderQueue() {
                {
                    UntexturedVertexArrayRenderingConfig config;
                    std::for_each( untexturedInstanceRenderQueue_.begin(), untexturedInstanceRenderQueue_.end(), RenderUntexturedInstances( &renderMeshLibrary_ ) );
                }                
                untexturedInstanceRenderQueue_.clear();
            }
            
            
            
            void renderTexturedInstanceRenderQueue() {             
                {
                    TexturedVertexArrayRenderingConfig config;
                    std::for_each( texturedInstanceRenderQueue_.begin(), texturedInstanceRenderQueue_.end(), RenderTexturedInstances( &renderMeshLibrary_, &textureIdLibrary_ ) );
                }
                
                texturedInstanceRenderQueue_.clear();
            }
            
            
            void renderText3d() {
                std::for_each( text3dRenderQueue_.begin(), text3dRenderQueue_.end(), RenderText3d() );
                text3dRenderQueue_.clear();
            }
            
            
            void renderText2d() {
                std::for_each( text2dRenderQueue_.begin(), text2dRenderQueue_.end(), RenderText2d() );
                text2dRenderQueue_.clear();
            }
            
            
            /**
             * Render all content in the render queues and clear the render
             * render qeues afterwards.
             *
             * Only renders real 3d content, not text.
             */
            void renderAll() {
                // renderLibraryRenderQueue();
                renderAllUntexturedUntransformedMeshes();
                renderUntexturedRenderQueue();
                renderTexturedRenderQueue();
                renderUntexturedInstanceRenderQueue();
                renderTexturedInstanceRenderQueue();
            }
            
            
            
            bool addTexture( SharedPointer< OpenGlTexture > const& _texture, TextureId& _id ) {
                // @todo Add a case for @c GL_TEXTURE_1D.
                SharedPointer< OpenGlTexture2dObject > texture( new OpenGlTexture2dObject( *_texture ) );
                
                if ( ! texture->valid() ) {
                    return false;
                }
                
                // Save id in list of texture ids.
                typedef TextureIdLibrary::iterator iterator;
                std::pair< iterator, bool > addResult = textureIdLibrary_.insert( std::make_pair( texture->id(), texture ) );
                if ( false == addResult.second ) {
                    return false;
                }
                
                // Put valid texture id into the return parameter.
                _id = texture->id();
                return true;
            }
            
            
            
            void removeTexture( TextureId const& _id ) {
                textureIdLibrary_.erase( _id );
            }
            
            
            bool containsTexture( TextureId const& _id ) const {
                return textureIdLibrary_.end() != textureIdLibrary_.find( _id );
            }
            
            
            bool validTexture( TextureId const& _id ) const {
            
                typedef TextureIdLibrary::const_iterator const_iterator;
                const_iterator iter = textureIdLibrary_.find( _id );
                
                if ( textureIdLibrary_.end() == iter ) {
                    return false;
                }
                
                return (*iter).second->valid();
            }
            
            
            void clearTextureLibrary() {
                textureIdLibrary_.clear();
            }
            
            
            

            
            
            // Meshes without own transformation matrices and without textures.
            OpenGlRenderMesh lines_;
            OpenGlRenderMesh triangles_;
            OpenGlRenderMesh points_;
            OpenGlRenderMesh quads_;
            
            // Meshes to render from the library.
            // typedef std::vector< TransformedLibraryMesh > TransformedLibraryMeshes;
            // TransformedLibraryMeshes libraryRenderQueue_;
            
            // Untextured meshes to render.
            TransformedMeshes untexturedRenderQueue_;
            
            // Textured meshes to render.
            TexturedTransformedMeshes texturedRenderQueue_;
            
            
            
            
            // Mesh library each mesh identified by its id.
            // typedef OpenGlRenderer::InstanceId RenderMeshLibraryId;
            // typedef std::map< RenderMeshLibraryId, SharedPointer< OpenGlRenderMesh > > RenderMeshLibrary;
            RenderMeshLibrary renderMeshLibrary_;
            RenderMeshLibraryId nextRenderMeshLibraryId_;
            
            
            // typedef std::map< TextureId, SharedPointer< OpenGlTexture2dObject >  > TextureIdLibrary;
            TextureIdLibrary textureIdLibrary_;
            
            
            // Stores the transformations of the instances to render to allow
            // speed optimizations if more than one instance is to be drawn by
            // not transmatting the render mesh geometry more than one to the
            // hardware.
            typedef std::map< RenderMeshLibraryId, std::vector< OpenSteer::Matrix > > InstanceRenderQueue;
            InstanceRenderQueue untexturedInstanceRenderQueue_;
            InstanceRenderQueue texturedInstanceRenderQueue_;
            
            // typedef std::pair< Vec3, OpenGlRenderText > Text3d;
            typedef std::vector< Text3d > Text3DRenderQueue;
            Text3DRenderQueue text3dRenderQueue_;
            
            typedef std::vector< Text2d > Text2dRenderQueue;
            Text2dRenderQueue text2dRenderQueue_;
        }; // struct OpenGlRenderer::Impl
        
        
    } // namespace Graphics
    
} // namespace OpenSteer




OpenSteer::Graphics::OpenGlRenderer::OpenGlRenderer() 
: renderStore_( new Impl()  )
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlRenderer::~OpenGlRenderer()
{
    // Nothing to do.
}



void 
OpenSteer::Graphics::OpenGlRenderer::render( Matrix const& _modelView, Matrix const& _projection )
{
	KAPAGA_UNUSED_PARAMETER( _modelView );
	KAPAGA_UNUSED_PARAMETER( _projection );
	
    // OpenGL has a right handed coordiante system with the thumb at the x-axis,
    // the index-finger is the y-axis and points up and the middle-finger is the
    // z-axis.
    // In default mode the camera sits at the origin an points down the negative 
    // z-axis (in the opposite direction of the middle-finger).
    
    // @todo Implement.
    // @todo Currently any modelview and projection specific handling is missing
    //       as this relies on the calls of OpenSteer.
        
    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    // glLoadMatrixf( _projection.data() );
    
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    // glLoadMatrixf( _modelView.data() );
    
    renderStore_->renderAll();
    
    // Restore the old projection and modelview matrices to eliminate undesired
    // side effects of manipulating the OpenGL state.
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();
}



void 
OpenSteer::Graphics::OpenGlRenderer::renderText( Matrix const& _modelView, Matrix const& _projection, float _width, float _height )
{
	KAPAGA_UNUSED_PARAMETER( _modelView );
	KAPAGA_UNUSED_PARAMETER( _projection );
	
    // OpenGL has a right handed coordiante system with the thumb at the x-axis,
    // the index-finger is the y-axis and points up and the middle-finger is the
    // z-axis.
    // In default mode the camera sits at the origin an points down the negative 
    // z-axis (in the opposite direction of the middle-finger).
    
    // @todo Implement. Don't use a projection matrix but arguments for the 
    //       width and heigth of the ortographic projection.
    // @todo Currently any modelview and projection specific handling is missing
    //       as this relies on the calls of OpenSteer.
    
    // Render text that is positioned in the 3d scene.
    glMatrixMode( GL_PROJECTION );
    
     /*
     glPushMatrix();
    
    glLoadIdentity();
    glOrtho( 0.0f, _width, 0.0f, _height, -1.0f, 1.0f );
    // glLoadMatrixf( _projection.data() );
     */
    
    glMatrixMode( GL_MODELVIEW );
    //glPushMatrix();
    // glLoadMatrixf( _modelView.data() );
    // glLoadIdentity();
    
    /*
    float modelViewMatrix[16];
    glGetFloatv( GL_MODELVIEW_MATRIX, modelViewMatrix );
    Matrix mv( &modelViewMatrix[ 0 ], &modelViewMatrix[ 16 ] );
    std::cout << mv << std::endl;
    */
    
    renderStore_->renderText3d();
    
    // Restore the old projection and modelview matrices to eliminate undesired
    // side effects of manipulating the OpenGL state.
    //glMatrixMode( GL_PROJECTION );
    //glPopMatrix();
    //glMatrixMode( GL_MODELVIEW );
    //glPopMatrix();
    
    // Render 2d HUD text.
    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();
    glOrtho( 0.0f, _width, 0.0f, _height, -1.0f, 1.0f );
    
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();
    
    renderStore_->renderText2d();
    
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
    
    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();
    
}



void 
OpenSteer::Graphics::OpenGlRenderer::precedeFrame()
{
    // @todo Implement.
    renderStore_->clearRenderQueues();
}



bool 
OpenSteer::Graphics::OpenGlRenderer::addToRenderMeshLibrary( SharedPointer< OpenGlRenderMesh > const& _mesh, 
                                                             InstanceId& _id )
{
    return renderStore_->addToRenderMeshLibrary( _mesh, _id );
}



void 
OpenSteer::Graphics::OpenGlRenderer::removeFromRenderMeshLibrary( InstanceId const& _id  )
{
    renderStore_->removeFromRenderMeshLibrary( _id );
}



bool 
OpenSteer::Graphics::OpenGlRenderer::inRenderMeshLibrary( InstanceId const& _id )
{
    return inRenderMeshLibrary( _id );
}



void 
OpenSteer::Graphics::OpenGlRenderer::clearRenderMeshLibrary()
{
    renderStore_->clearRenderMeshLibrary();
}




void 
OpenSteer::Graphics::OpenGlRenderer::addToRender( Matrix const& _transformation, 
                                                  SharedPointer< OpenGlRenderMesh > const& _primitive )
{
    renderStore_->add( _transformation, _primitive );
    
}



void 
OpenSteer::Graphics::OpenGlRenderer::addToRender( SharedPointer< OpenGlRenderMesh > const& _primitive )
{
    renderStore_->add( _primitive );
}



void 
OpenSteer::Graphics::OpenGlRenderer::addToRender( Matrix const& _transformation, 
                                                  InstanceId const& _id )
{
    renderStore_->add( _transformation, _id );
}



void 
OpenSteer::Graphics::OpenGlRenderer::addToRender( InstanceId const& _id )
{
    renderStore_->add( OpenSteer::identityMatrix, _id );    
}



void 
OpenSteer::Graphics::OpenGlRenderer::addTextToRender( Matrix const& _transformation,
                                                      SharedPointer< OpenGlRenderText > const& _text )
{
    renderStore_->addText( _transformation, _text );
}


void 
OpenSteer::Graphics::OpenGlRenderer::addText2dToRender( SharedPointer< OpenGlRenderText2d > const& _text )
{
    renderStore_->addText2d( _text );
}



void 
OpenSteer::Graphics::OpenGlRenderer::clear()
{
    // @todo Should this also clear the texture library? Seemingly not, 
    //       otherwise it would also clear the render mesh library.
    renderStore_->clearRenderQueues();
}


bool 
OpenSteer::Graphics::OpenGlRenderer::addTexture( SharedPointer< OpenGlTexture > const& _texture, TextureId& _id )
{
    return renderStore_->addTexture( _texture, _id );
}


void 
OpenSteer::Graphics::OpenGlRenderer::removeTexture( TextureId const& _id )
{
    renderStore_->removeTexture( _id );
}


bool 
OpenSteer::Graphics::OpenGlRenderer::containsTexture( TextureId const& _id ) const
{
    return renderStore_->containsTexture( _id );
}


