#ifndef OPENSTEER_GRAPHICS_OPENGLRENDERER_H
#define OPENSTEER_GRAPHICS_OPENGLRENDERER_H

// Include std::auto_ptr
#include <memory>



// Include OpenSteer::Graphics::Renderer
#include "OpenSteer/Graphics/RenderService.h"

// Include OpenSteer::Graphics::OpenGlTexture
#include "OpenSteer/Graphics/OpenGlTexture.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"


namespace OpenSteer {
    
    // Forward declaration.
    class Matrix;
    
    
    namespace Graphics {
        
        
        // Forward declarations.
        class OpenGlRenderText;
        class OpenGlRenderText2d;
        class OpenGlRenderMesh;
        class OpenGlTexture;
        
        
        /**
         * @todo Add an output iterator.
         * @todo Make adding meshes to the renderer more stream-like. Only the
         *       library should stay non-stream-alike.
         * @todo Don't add directly to a renderer but use a frame-batch or
         *       frame-chunk that gets configured in a special way and is then
         *       feed into the renderer. This way global state changes are
         *       nicely encapsulated.
         */
        class OpenGlRenderer : public Renderer {
        public:
            
            /**
            * Might be independent of @c RenderFeeder::InstanceId!
             */
            typedef unsigned int InstanceId;
            typedef unsigned int TextureId;
            
            OpenGlRenderer();
            
            virtual ~OpenGlRenderer();
            
            /**
             * Only renders the 3d content, not text.
             * @todo Rework the renderer to use a frame object to render so
             * modelview and projection matrices aren't needed in this call but
             * are provided in the frame object.
             */
            virtual void render( Matrix const& _modelView, Matrix const& _projection );
            
            /**
             * Only renders the text (HUD). 2d text only uses the projection
             * matrix and the positions stored assigned to it by the 
             * @c addText2d member function while 3d positioned text also needs 
             * the @a _modelView information to show at the right position in 
             * the 3d scene.
             * @todo Remove dependency on GLUT.
             */
            virtual void renderText( Matrix const& _modelView, Matrix const& _projection, float _width, float _height );
            
            virtual void precedeFrame();
            
            bool addToRenderMeshLibrary( SharedPointer< OpenGlRenderMesh > const& _mesh, 
                                              InstanceId& _id );
            void removeFromRenderMeshLibrary( InstanceId const& _id  );
            bool inRenderMeshLibrary( InstanceId const& _id );
            
            /**
             * If called while render meshes from the library have been inserted
             * into the render queues these meshes will be drawn with the next 
             * @c render call.
             */
            void clearRenderMeshLibrary();
            
            void addToRender( Matrix const& _transformation, 
                              SharedPointer< OpenGlRenderMesh > const& _mesh );
            void addToRender( SharedPointer< OpenGlRenderMesh > const& _mesh );
            void addToRender( Matrix const& _transformation, 
                              InstanceId const& _id );
            void addToRender( InstanceId const& _id );
            
            
            void addTextToRender( Matrix const& _transformation,
                                  SharedPointer< OpenGlRenderText > const& _text );
            void addText2dToRender( SharedPointer< OpenGlRenderText2d > const& _text );
            // void addTextToRender( SharedPointer< OpenGlRenderText2d > const& _text );
            
            /**
             * Clears all added meshes or instances to draw that haven't been
             * rendered. The instance library and the texture library aren't
             * touched.
             */
            void clear();
            
            /**
             * Adds a new texture @a _texture to the renderer and returns the 
             * id of it in @a _id. If the texture has been added to the renderer
             * @c true is returned, otherwise @c false. In the last case @a _id
             * shouldn't be used.
             *
             * To change a texture it needs to be removed and then added again
             * with the changes applied.
             *
             * @todo Would a better interface be possible if textures are 
             *       created by explicit interface functions instead of by 
             *       instantiating them outside of the renderer?
             */
            bool addTexture( SharedPointer< OpenGlTexture > const& _texture, TextureId& _id );
            
            /**
             * Removes the texture associated with @a _id from the renderer
             * if it is contained.
             *
             * Don't remove a texture as long as it is in use by a mesh or an 
             * instance id stored in the renderer otherwise behavior is 
             * undefined.
             */
            void removeTexture( TextureId const& _id );
            
            /**
             * Returns @c true if a texture associated with @a _id is contained,
             * @c false otherwise.
             */
            bool containsTexture( TextureId const& _id ) const;
            
            
        private:
            /**
             * Intentionally not implemented to prevent copies.
             */
            OpenGlRenderer( OpenGlRenderer const& );
            
            /**
             * Intentionally not implemented to prevent assignment.
             */
            OpenGlRenderer& operator=( OpenGlRenderer );
            
        private:
                
            struct Impl;
            std::auto_ptr< Impl > renderStore_;
            
            
        }; // class OpenGlRenderer
        
        
    } // namespace Graphics
    
    
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_OPENGLRENDERER_H

