#ifndef OPENSTEER_GRAPHICS_OPENGLRENDERMESH_H
#define OPENSTEER_GRAPHICS_OPENGLRENDERMESH_H

// Include std::vector
#include <vector>


// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenGL, glut, glu
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Include OpenSteer::Vec3
#include "OpenSteer/Vec3.h"

namespace OpenSteer {
    
    
    namespace Graphics {
        
        
        struct OpenGlRenderMesh {
            
            typedef size_t size_type;
            typedef GLint TextureId;
            
            enum Type { POINTS = GL_POINTS, 
                        LINES = GL_LINES, 
                        LINE_STRIP = GL_LINE_STRIP, 
                        LINE_LOOP = GL_LINE_LOOP, 
                        TRIANGLES = GL_TRIANGLES, 
                        TRIANGLE_STRIP = GL_TRIANGLE_STRIP, 
                        TRIANGLE_FAN = GL_TRIANGLE_FAN, 
                        QUADS = GL_QUADS, 
                        POLYGON = GL_POLYGON };
            
            enum TextureFunction { DECAL = GL_DECAL, 
                                   REPLACE = GL_REPLACE,
                                   MODULATE = GL_MODULATE,
                                   BLEND = GL_BLEND };
            
            
            static size_type const vertexElementCount_ = 3;
            static size_type const vertexStride_ = 0;
            
            static size_type const colorElementCount_ = 4;
            static size_type const colorStride_ = 0;
            
            static size_type const normalElementCount_ = 3;
            static size_type const normalStride_ = 0;
            
            static size_type const textureCoordinateElementCount_ = 2;
            static size_type const textureCoordinateStride_ = 0;
            
            
            
            explicit OpenGlRenderMesh( Type _type, size_type _meshVertexIndexCount = 0, TextureFunction _textureFunction = REPLACE ) 
                : vertices_( 0 ), 
                colors_( 0 ), 
                normals_( 0 ), 
                textureCoordinates_( 0 ), 
                indices_( 0 ), 
                textureId_(),
                textureFunction_( _textureFunction ),
                type_( _type ) {
                
                    vertices_.reserve( _meshVertexIndexCount * vertexElementCount_ );
                    colors_.reserve( _meshVertexIndexCount * colorElementCount_ );
                    normals_.reserve( _meshVertexIndexCount * normalElementCount_ );
                    textureCoordinates_.reserve( _meshVertexIndexCount * textureCoordinateElementCount_ );
                    indices_.reserve( _meshVertexIndexCount );
                    
                
                }
            
            
            void clear();
            void shrinkContainersToFit();
            void clearAndShrinkContainersToFit();
            
            
            /**
                * Three floats per vertex.
             */
            std::vector< GLfloat > vertices_;
            
            /**
                * RGBA.
             * Four floats per color per vertex.
             */
            std::vector< GLfloat > colors_;
            
            /**
                * Three floats per normal per vertex.
             */
            std::vector< GLfloat > normals_;
            
            /**
                * Two floats per coordinate per vertex.
             */
            std::vector< GLfloat > textureCoordinates_;
            
            /**
             * One index per primitive vertex.
             */
            std::vector< GLuint > indices_;
            
            /**
             * If @c textureId_ is @c 0 no textures are used. OpenGL provides 
             * a default texture if it is used nonetheless.
             */
            TextureId textureId_;
            
            TextureFunction textureFunction_;
            
            Type type_;
            
        }; // struct OpenGlRenderMesh
        
        /**
         * @todo Remove.
         */
        void print( OpenGlRenderMesh const& primitive );
        
        inline bool textured( OpenGlRenderMesh const& _mesh ) {
            // An OpenGL texture id of @c 0 descibes the default texture - which is
            // interpreted as if @a _mesh is untextured.
            return 0 != _mesh.textureId_;
        }
        
        inline void insertAtBack( std::vector< float >& _container, Vec3 const& _data ) {
            _container.insert( _container.end(), _data.data(), _data.data() + 3 );
        }
        
        inline void insertAtBack( std::vector< float >& _colors, Color const& _color ) {
            _colors.insert( _colors.end(), _color.colorFloatArray(), _color.colorFloatArray() + OpenGlRenderMesh::colorElementCount_ );
        }
        
        /**
            * Adds the data of @a _source to @a _target.
         *
         * Adapts the indices added to the @a _target data.
         *
         * Doesn't check if the types of @a _target and @a _source are
         * identical or if they use the same texture. Call @c mergeMeshes for 
         * additional checks.
         */
        void mergeMeshesFast( OpenGlRenderMesh& _target, OpenGlRenderMesh const& _source );
        
        /**
            * Adds the data of @a _source to @a _target.
         *
         * Adapts the indices added to the @a _target data.
         *
         * @a _source data is only added to @a _target if both have the same
         * @c type_ and @c textureId_. Afterwards @c mergeMeshesFast is called.
         */
        bool mergeMeshes( OpenGlRenderMesh& _target, OpenGlRenderMesh const& _source );
        
    } // namespace Graphics
    
    
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_OPENGLRENDERMESH_H
