#ifndef OPENSTEER_GRAPHICS_LOCKINGRENDERFEEDER_H
#define OPENSTEER_GRAPHICS_LOCKINGRENDERFEEDER_H

// Include OpenSteer::Graphics::RenderFeeder, OpenSteer::Graphics::NullRenderFeeder
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"


namespace OpenSteer {	
namespace Graphics {
		
	/**
	 * Render feeder that protects the access to an associated render feeder though a critical 
	 * section or a lock to avoid concurrent access and therefore race conditions.
	 * As long as threads are calling the render or graphics library adding and removing 
	 * member functions no other member function must be called - especially the protected render
	 * feeder mustn't be touched by other functions then. Or to state it even clearer:
	 *
	 * @ATTENTION The only member functions that can be called concurrently are the @c render and
	 *            the graphics library changing member functions. All other member functions aren't
	 *            thread-safe and should be called only from sequential code!
	 */
	class LockingRenderFeeder : public RenderFeeder {
	public:
		
		LockingRenderFeeder();
		
		explicit LockingRenderFeeder( SharedPointer< RenderFeeder > const& _targetFeeder );
		
		LockingRenderFeeder( LockingRenderFeeder const& other );
		
		virtual ~LockingRenderFeeder();
		
		LockingRenderFeeder& operator=( LockingRenderFeeder other );
		
		void swap( LockingRenderFeeder& other );
		
		
		void setRenderFeederToLock( SharedPointer< RenderFeeder > const& renderFeederToLock );
		SharedPointer< RenderFeeder > renderFeederToLock() const;
		
		virtual void render( Matrix const& _matrix, InstanceId const& _instanceId );
		virtual void render( InstanceId const& _instanceId );
		virtual void render( Matrix const& _matrix, GraphicsPrimitive const& _graphicsPrimitive );
		virtual void render( GraphicsPrimitive const& _graphicsPrimitive );
		
		virtual bool addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _graphicsPrimitive, InstanceId& _instanceId );
		virtual void removeFromGraphicsPrimitiveLibrary( InstanceId const& _instanceId );
		virtual bool inGraphicsPrimitiveLibrary( InstanceId const& _instanceId ) const;
		virtual void clearGraphicsPrimitiveLibrary();
		
	private:
		SharedPointer< RenderFeeder > renderFeederToLock_;
		
	}; // class LockingRenderFeeder
	
	
} // namespace OpenSteer
} // namespace Graphics




#endif // OPENSTEER_GRAPHICS_LOCKINGRENDERFEEDER_H
