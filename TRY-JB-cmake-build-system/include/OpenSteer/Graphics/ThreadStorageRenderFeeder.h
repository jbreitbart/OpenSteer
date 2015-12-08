#ifndef OPENSTEER_GRAPHICS_THREADSTORAGERENDERFEEDER_H
#define OPENSTEER_GRAPHICS_THREADSTORAGERENDERFEEDER_H

// Include OpenSteer::Graphics::RenderFeeder, OpenSteer::Graphics::NullRenderFeeder
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::ThreadStorage
#include "kapaga/thread_storage.h"



namespace OpenSteer {	
	namespace Graphics {
		
		/**
		 * Render feeder that hands calls to its render and graphics library member functions to 
		 * a render feeder stored in a slot of so called <em>thread storage</em>.
		 *
		 * The slot is determined by evaluating the OpenMP thread id of the calling thread.
		 * The render feeder itself does not do any locking or other synchronization. 
		 *
		 * @ATTENTION If more than one thread can have the same OpenMP thread id this render feeder 
		 *            isn't safe to use!
		 *
		 * @todo Write a special class for thread storage instead of using a @c std::vector. This 
		 *       class should contain debug assertions to detect wrong usage and race conditions.
		 */
		class ThreadStorageRenderFeeder : public RenderFeeder {
		public:
			
			typedef kapaga::thread_storage< SharedPointer< RenderFeeder > > ThreadStorageType;
			
			ThreadStorageRenderFeeder();
			
			explicit ThreadStorageRenderFeeder( SharedPointer< ThreadStorageType > const& _threadStorage );
			
			ThreadStorageRenderFeeder( ThreadStorageRenderFeeder const& other );
			
			virtual ~ThreadStorageRenderFeeder();
			
			ThreadStorageRenderFeeder& operator=( ThreadStorageRenderFeeder other );
			
			void swap( ThreadStorageRenderFeeder& other );
			
			/**
			 * Sets a different thread storage to use.
			 *
			 * Don't pass in a shared pointer to a null pointer as no checks are done to catch 
			 * such a situation and behavior isn't defined then.
			 */
			void setThreadStorage(  SharedPointer< ThreadStorageType > const& _threadStorage );
			
			// @todo Remove to prevent access to instance internals?
			SharedPointer< ThreadStorageType > const& threadStorage() const;
			
			/**
			 * Calls @c render on the slot of thread storage associated with the calling thread.
			 */
			virtual void render( Matrix const& _matrix, InstanceId const& _instanceId );
			
			/**
			 * Calls @c render on the slot of thread storage associated with the calling thread.
			 */
			virtual void render( InstanceId const& _instanceId );
			
			/**
			 * Calls @c render on the slot of thread storage associated with the calling thread.
			 */
			virtual void render( Matrix const& _matrix, GraphicsPrimitive const& _graphicsPrimitive );
			
			/**
			 * Calls @c render on the slot of thread storage associated with the calling thread.
			 */
			virtual void render( GraphicsPrimitive const& _graphicsPrimitive );
			
			/**
			 * @todo Check if the behavior in sequential code is ok.
			 *
			 * @attention The member function isn't thread safe if the render feeder used in the
			 *            thread storage isn't thread safe. Don't call in a parallel region if in
			 *            doubt!
			 * @attention If called in a sequential region of code only the first render feeder of
			 *            the thread storage is called. The others don't try to register the 
			 *            instance. If this behavior is needed it has to be done by hand.
			 * @attention If called from a parallel region of code then for every thread the 
			 *            associated render feeder is called. This might not be what is intended
			 *            if all render feeder feed the same renderer. It might be the best no to 
			 *            call this member function in a parallel region.
			 */
			virtual bool addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _graphicsPrimitive, InstanceId& _instanceId );
			
			/**
			 * @todo Check if the behavior in sequential code is ok.
			 *
			 * @attention The member function isn't thread safe if the render feeder used in the
			 *            thread storage isn't thread safe. Don't call in a parallel region if in
			 *            doubt!
			 * @attention If called in a sequential region of code only the first render feeder of
			 *            the thread storage is called. The others don't try to register the 
			 *            instance. If this behavior is needed it has to be done by hand.
			 * @attention If called from a parallel region of code then for every thread the 
			 *            associated render feeder is called. This might not be what is intended
			 *            if all render feeder feed the same renderer. It might be the best no to 
			 *            call this member function in a parallel region.
			 */
			virtual void removeFromGraphicsPrimitiveLibrary( InstanceId const& _instanceId );
			
			/**
			 * @todo Check if the behavior in sequential code is ok.
			 *
			 * @attention The member function isn't thread safe if the render feeder used in the
			 *            thread storage isn't thread safe. Don't call in a parallel region if in
			 *            doubt!
			 * @attention If called in a sequential region of code only the first render feeder of
			 *            the thread storage is called. The others don't try to register the 
			 *            instance. If this behavior is needed it has to be done by hand.
			 * @attention If called from a parallel region of code then for every thread the 
			 *            associated render feeder is called. This might not be what is intended
			 *            if all render feeder feed the same renderer. It might be the best no to 
			 *            call this member function in a parallel region.
			 */
			virtual bool inGraphicsPrimitiveLibrary( InstanceId const& _instanceId ) const;
			
			/**
			 * @todo Check if the behavior in sequential code is ok.
			 *
			 * @attention The member function isn't thread safe if the render feeder used in the
			 *            thread storage isn't thread safe. Don't call in a parallel region if in
			 *            doubt!
			 * @attention If called in a sequential region of code only the first render feeder of
			 *            the thread storage is called. The others don't try to register the 
			 *            instance. If this behavior is needed it has to be done by hand.
			 * @attention If called from a parallel region of code then for every thread the 
			 *            associated render feeder is called. This might not be what is intended
			 *            if all render feeder feed the same renderer. It might be the best no to 
			 *            call this member function in a parallel region.
			 */
			virtual void clearGraphicsPrimitiveLibrary();
			
		private:
			 SharedPointer< ThreadStorageType > threadStorage_;
			
		}; // class ThreadStorageRenderFeeder
		
		
		/**
		 * Makes a thread storage containing shared pointers of individual copies of the specified
		 * render feeder type for every slot.
		 */
		template< class RenderFeederType >
		SharedPointer< ThreadStorageRenderFeeder::ThreadStorageType > makeThreadStorage( RenderFeeder::size_type slot_count, RenderFeederType const& renderFeederToDuplicateForThreadStoareSlots );
		
		
	} // namespace Graphics
} // namespace OpenSteer




template< class RenderFeederType >
OpenSteer::SharedPointer< OpenSteer::Graphics::ThreadStorageRenderFeeder::ThreadStorageType > OpenSteer::Graphics::makeThreadStorage( RenderFeeder::size_type slot_count,
											RenderFeederType const& renderFeederToDuplicateForThreadStoareSlots )
{
	using namespace OpenSteer;
	using namespace OpenSteer::Graphics;
	
	SharedPointer< ThreadStorageRenderFeeder::ThreadStorageType > result( new ThreadStorageRenderFeeder::ThreadStorageType( slot_count ) );

	for ( ThreadStorageRenderFeeder::ThreadStorageType::size_type i = 0; i < slot_count; ++i ) {
		result->accessSlotSequentially( i ).reset( new RenderFeederType( renderFeederToDuplicateForThreadStoareSlots ) );
	}
	
	
	return result;
}




#endif // OPENSTEER_GRAPHICS_THREADSTORAGERENDERFEEDER_H
