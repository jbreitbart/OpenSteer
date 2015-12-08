#ifndef OPENSTEER_GRAPHICS_RENDERFEEDER_H
#define OPENSTEER_GRAPHICS_RENDERFEEDER_H

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::Graphics::Primitive
#include "OpenSteer/Graphics/GraphicsPrimitives.h"


namespace OpenSteer {

    namespace Graphics {
    
        
        
        /**
         * Mixin interface to enable reuse of member function definitions.
         *
         * Because it is planned to be used as a mixin the member function
         * names have more descriptive names than it would be appropriate if
         * the class would be used to inherit pure libraries from it.
         *
         * @todo Use exceptions?
         */
        class GraphicsPrimitiveLibraryMixin {
        public:
            typedef size_t InstanceId;
            
            virtual ~GraphicsPrimitiveLibraryMixin() {};
            
            virtual bool addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _primitive, InstanceId& _id ) = 0;
            // virtual bool addToGraphicsPrimitiveLibrary( Matrix const& _transformation, GraphicsPrimitive const& _primitive, InstanceId& _id ) = 0;
            virtual void removeFromGraphicsPrimitiveLibrary( InstanceId const& _id ) = 0;
            virtual bool inGraphicsPrimitiveLibrary( InstanceId const& _id ) const = 0;
            virtual void clearGraphicsPrimitiveLibrary() = 0;
            
        }; // class GraphicsPrimitiveLibrary
        
        
        /**
         * Instances that comply to the @c RenderFeeder interface take graphics
         * primitives to draw and feed them into the associated renderer in a
         * render specific appropriate fashion, for exmaple to optimize render
         * performance.
         *
         * An implementation of a @c RenderFeeder could be a full blown scene
         * graph or logic to feed or stream data to a specific renderer
         * implementation. It has to know if it is better to dump everything
         * that has been inserted at once into the renderer or to divide it into
         * batches that are feed into the renderer from time to time. 
         *
         * It has to care to feed all data belonging to a specific frame to 
         * render just into the renderer for the specific frame and not for 
         * another frame.
         *
         * As the @c RenderFeeder has no idea about the camera or the projection
         * settings of the renderer the programmer has to care about this
         * settings and the right order of configuring and feeding the renderer.
         */ 
        class RenderFeeder : public GraphicsPrimitiveLibraryMixin {
        public:
            typedef size_t size_type;
            
            virtual ~RenderFeeder() {};
            
            virtual void render( Matrix const& _transformation, InstanceId const& _instanceId ) = 0;
            virtual void render( InstanceId const& _instanceId ) = 0;
            virtual void render( Matrix const& _transformation, GraphicsPrimitive const& _primitive ) = 0;
            virtual void render( GraphicsPrimitive const& _primitive ) = 0;
            
        }; // class RenderFeeder
        
        
        
        
        /**
         * Dummy render feeder just doing nothing.
         *
         * It is used if annotation logging is disabled by setting the 
         * render feeder to @c 0.
         */
        class NullRenderFeeder : public RenderFeeder {
        public:
            virtual ~NullRenderFeeder() {}
            
            virtual void render( Matrix const& , InstanceId const& ) {}
            virtual void render( InstanceId const& ) {}
            virtual void render( Matrix const& , GraphicsPrimitive const&  ) {}
            virtual void render( GraphicsPrimitive const&  ) {}
            
            virtual bool addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& , InstanceId&  ) { return false; }
            // virtual bool addToGraphicsPrimitiveLibrary( Matrix const& , GraphicsPrimitive const& , InstanceId&  ) { return false; }
            virtual void removeFromGraphicsPrimitiveLibrary( InstanceId const&  ) {}
            virtual bool inGraphicsPrimitiveLibrary( InstanceId const&  ) const { return false; }
            virtual void clearGraphicsPrimitiveLibrary() {}
        }; // class NullRenderFeeder


    } // namespace Graphics

} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_RENDERFEEDER_H
