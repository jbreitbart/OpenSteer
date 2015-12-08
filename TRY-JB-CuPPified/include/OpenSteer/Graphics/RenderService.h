/**


- July 13, 2005
-- There are two ways I am inspecting how to write a renderer that:
--- is feed with so called graphics primitives that are non-render specific representations for certain more or less complex graphics objects to render
--- the renderer has to map theses primitives to its functionality how to draw it using an interface that doesn't directly detect the full type of a graphics primitive but just that is is inherited from an abstract base class called GraphicsPrimitive.
-- One way is to add sort of a visitor concept to the graphics primitives. If a graphics primitive is entered into the renderer via a base class pointer or reference the renderer calls an "identify" member function of the primitive and inserts itself as its argument. The specific primitive calls a function of the renderer passing "*this" or "this" as its argument. If the renderer has overloaded this function with the specific type it detects the real type and can react based on it. If the type is unknown a function with the base class type argument is called which just says: "unknown type".

-- The other way is that every inherited renderer has an associated map that links graphics primitives with specific render primitives or functions that translate the graphics primitive into render primitives that just this specific renderer understands. This could also be done in the render feeder and not the renderer itself.

--- If a graphics primitive is entered into the renderer or a render feeder it takes the typinfo of the graphics primitive and looks into a registry table how to translate it into a render primitive the renderer understands. This render primitive is then injected into the renderer and the renderer can do with it whatever it wants. 

--- To make this fool proof there needs to be a render service for each specific renderer which delivers you with render feeders and the renderer itself and which is used to add further mappings.

---> Write a translator object that registers type descriptors with translator functors that might directly feed their translations into the renderer or whatever (they don't have a return type but only "eat" a graphics primitive). Think about writing this as a template solution to use compile time optimizations to really speed this up a bit. Have to experiment with this to see if the performance would be ok even if this would be completely run-time (think plugin and dynamic library) capable.

--- Think multiple/double dispatching.

-- By the way, the main problem stems from the fact that I want to use a pointer of a base type to insert graphics primitives into the render feeder and don't want to add member functions for every graphics primitive to it because this would result in an explosion of member functions to write, for each type one without a transformation matrix, with a transformation matrix, one with and without a transformation matrix to store the graphics primitive in a library and so on. This will just get worser the moment specific render states (like "wireframe") are added.

-> Conclusion: use the second approach to prevent the explosion of member functions to write!

*/













#ifndef OPENSTEER_GRAPHICS_RENDERSERVICE_H
#define OPENSTEER_GRAPHICS_RENDERSERVICE_H

// Include std::type_info
// #include <typeinfo>




// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"



namespace OpenSteer {

    
    // Forward declaration.
    class Matrix;
    
    namespace Graphics {
        
        // Forward declaration.
        class RenderFeeder;

        
        
        /**
         * A renderer implementing the @c RenderFeeder interface itself to 
         * draw graphics. Inherit from it to provide specific renderer.
         *
         * @todo Camera and projection must be set - belongs into 
         *       @c RenderFeeder?
         */
        class Renderer {
        public:
            
            virtual ~Renderer() {};
            
            /**
             * Draw the graphical primitives stored so far into the current
             * frame and remove them.
             *
             * @todo Take matrices or special objects?
             */
            virtual void render( Matrix const& _modelView, Matrix const& _projection ) = 0;
            
            /**
             * Only renders the text (HUD). 2d text only uses the projection
             * matrix and the positions stored assigned to it by the 
             * @c addText2d member function while 3d positioned text also needs 
             * the @a _modelView information to show at the right position in 
             * the 3d scene.
             */
            virtual void renderText( Matrix const& _modelView, Matrix const& _projection, float _width, float _height ) = 0;
            
            /**
             * Show the current frame and prepare the next one.
             *
             * If primitives to draw have been injected into the renderer but
             * @c render hasn't been called the content won't be shown and will
             * be lost.
             *
             * This doesn't call the low-level graphics API flush or sync 
             * functions but just clears the internal structures.
             */
            virtual void precedeFrame() = 0;
            
            
        }; // class Renderer
        
        
        /**
         * Collection of renderer, render feeder, eventually geometry model 
         * loader, and graphics primitives to render primitives translators.
         *
         * To use it first create an inherited render service. Create a 
         * renderer and then create a render feeder for this renderer.
         *
         * To be able to feed graphics primitives into the renderer register
         * translators with the render service translator.
         *
         * @todo How does this fit into a distributed environment, for example
         *       using MPI? Remote render services could be used that link to 
         *       real render services on a specific node in the distributed
         *       system.
         */
        class RenderService {
        public:
            
            virtual ~RenderService() {}
            
            virtual  SharedPointer< Renderer > createRenderer() const = 0;
            virtual  SharedPointer< RenderFeeder > createRenderFeeder( SharedPointer< Renderer > const&  _renderer ) const = 0;
            
            // virtual void addTranslator( std::type_info const* _typeInfo, GraphicsPrimitiveTranslator const& _translator ) = 0;
            // virtual void removeTranslator( std::type_info const* _typeInfo ) = 0;
            // virtual bool contains(  std::type_info const* _typeInfo ) const = 0;
            // virtual void clearTranslators() = 0;
            
        }; // class RenderService
        
        
        
    } // namespace Graphics
    
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_RENDERSERVICE_H

