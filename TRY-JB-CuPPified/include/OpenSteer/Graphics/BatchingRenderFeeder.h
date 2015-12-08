#ifndef OPENSTEER_GRAPHICS_BATCHINGRENDERFEEDER_H
#define OPENSTEER_GRAPHICS_BATCHINGRENDERFEEDER_H

// Include std::pair
#include <utility>

// Include std::vector
#include <vector>



// Include OpenSteer::Graphics::RenderFeeder, OpenSteer::Graphics::NullRenderFeeder
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"


namespace OpenSteer {

    namespace Graphics {
    
        /**
         * Render feeder collecting graphics primitives and batches them all
         * together when calling @c batch to an associated render feeder.
         *
         * Only calls to add or remove graphics primitives to the library are
         * handled immediately.
         *
         * @todo Add a template future class to also implement the library 
         *       functionality asynchronously.
         *
         * @todo Add a function to every render feeder like @c flush so no 
         *       special @c batch function is needed and all render feeders 
         *       can be treated as equal.
         *
         */
        class BatchingRenderFeeder : public RenderFeeder {
        public:
            
            /**
             * Constructs the render feeder and sets a null render feeder to 
             * batch to.
             */
            BatchingRenderFeeder();
            
            /**
             * Constructs the render feeder and sets the feeder to batch to. 
             * If @a _renderFeederToBatchTo doesn't point to a valid render 
             * feeder a null render feeder is created internally.
             */
            explicit BatchingRenderFeeder( SharedPointer< RenderFeeder > const& _targetFeeder );
            
            /**
             * Constructs a render feeder copying the batch hold by @a _other
             * which feeds the same successing render feeder as @a _other.
             */
            BatchingRenderFeeder( BatchingRenderFeeder const& _other );
            virtual ~BatchingRenderFeeder();
            
            BatchingRenderFeeder& operator=( BatchingRenderFeeder _other );
            
            void swap( BatchingRenderFeeder& _other );
            
            /**
             * Sets the render feeder to batch to. If @a _renderFeederToBatchTo 
             * doesn't point to a valid render feeder a null render feeder is
             * created internally.
             */
            void setRenderFeederToBatchTo( SharedPointer< RenderFeeder > const& _renderFeederToBatchTo );
            SharedPointer< RenderFeeder > renderFeederToBatchTo() const;
            
            virtual void render( Matrix const& _matrix, InstanceId const& _instanceId );
            virtual void render( InstanceId const& _instanceId );
            virtual void render( Matrix const& _matrix, GraphicsPrimitive const& _graphicsPrimitive );
            virtual void render( GraphicsPrimitive const& _graphicsPrimitive );
            
            virtual bool addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _graphicsPrimitive, InstanceId& _instanceId );
            virtual void removeFromGraphicsPrimitiveLibrary( InstanceId const& _instanceId );
            virtual bool inGraphicsPrimitiveLibrary( InstanceId const& _instanceId ) const;
            virtual void clearGraphicsPrimitiveLibrary();
            
            
            void batch();
            
        private:
            typedef std::pair< Matrix, InstanceId > TransformedInstance;
            typedef std::pair< Matrix, SharedPointer< GraphicsPrimitive > > TransformedGraphicsPrimitive;
            
            std::vector< TransformedInstance > transformedInstancesToRender_;
            std::vector< InstanceId > instancesToRender_;
            std::vector< TransformedGraphicsPrimitive > transformedGraphicsPrimitivesToRender_;
            std::vector< SharedPointer< GraphicsPrimitive > > graphicsPrimitivesToRender_;
            
            SharedPointer< RenderFeeder > renderFeederToBatchTo_;
            
        }; // class BatchingRenderFeeder


    } // namespace Graphics

} // namespace OpenSteer

#endif // OPENSTEER_GRAPHICS_BATCHINGRENDERFEEDER_H
