#ifndef OPENSTEER_GRAPHICS_GRAPHICSANNOTATIONLOGGER_H
#define OPENSTEER_GRAPHICS_GRAPHICSANNOTATIONLOGGER_H

// Include OpenSteer::Graphics::RenderFeeder
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::Graphics::GraphicsPrimitive
#include "OpenSteer/Graphics/GraphicsPrimitives.h"

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"


namespace OpenSteer {
    
    
    namespace Graphics {
        
        /**
         * Wrapper around a @c RenderFeeder to allow easy exchange of 
         * the used render feeder at runtime, for example use a dummy or null
         * render feeder implementation to disable logging of graphics 
         * annotations. Also put commands to feed graphics into the logger
         * inside macros to make it easy at compile time to completely disable
         * the logging if it is just used for debugging purposes.
         *
         * @todo Add a @a finishedAnnotationForCurrentTick member function?
         *
         * It doesn't take over memory management for the @c RenderFeeder.
         */
        class GraphicsAnnotationLogger {
        public:    
            
            GraphicsAnnotationLogger(); 
            explicit GraphicsAnnotationLogger( RenderFeeder* _renderFeeder );
            
            /**
             * Set the render feeder to feed with annotation graphics.
			 * 
			 * Doesn't take over memory management of @a _renderFeeder.
             * Set @a _renderFeeder to @c 0 to disable logging.
             */
            void setRenderFeeder( RenderFeeder* _renderFeeder );
                  
            /**
             * 
             */
            void log( Matrix const& _transformation, GraphicsPrimitive const& _primitive ) {
                renderFeeder_->render( _transformation, _primitive );
            }
            
            void log( GraphicsPrimitive const& _primitive ) {
                renderFeeder_->render( _primitive );
            }
            
        private:
                
            /**
             * Intentionally not implemented to disable copying.
             */
            GraphicsAnnotationLogger( GraphicsAnnotationLogger const& );
            
            /**
             * Intentionally not implemented to disable assignment.
             */
            GraphicsAnnotationLogger& operator=( GraphicsAnnotationLogger );
            
        private:
                
            NullRenderFeeder nullRenderFeeder_;
            RenderFeeder* renderFeeder_;
        }; // class GraphicsAnnotationLogger
        
        
    } // namespace Graphics
    
} // namespace OpenSteer



#endif // OPENSTEER_GRAPHICS_GRAPHICSANNOTATIONLOGGER_H
