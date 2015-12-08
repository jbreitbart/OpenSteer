#include "OpenSteer/Graphics/LockingRenderFeeder.h"

// @todo Profile the code to see if reader-writer-locks might be more optimal instead of just one big lock.
// @todo Add a name to the critical sections. bknafla: I removed the name after getting compilers 
//       errors stating that the symbol of the name I used (LockingRenderFeeder) does not exist...

OpenSteer::Graphics::LockingRenderFeeder::LockingRenderFeeder()
: renderFeederToLock_( new NullRenderFeeder() )
{
	// Nothing to do.
}



OpenSteer::Graphics::LockingRenderFeeder::LockingRenderFeeder( SharedPointer< RenderFeeder > const& targetFeeder )
:renderFeederToLock_( 0 != targetFeeder ? targetFeeder : SharedPointer< RenderFeeder >( new NullRenderFeeder() ) )
{
	// Nothing to do.
}


// bknafla: @todo Check if it is necessary to call the copy constructor of a purely virtual base class.
OpenSteer::Graphics::LockingRenderFeeder::LockingRenderFeeder( LockingRenderFeeder const& other )
: RenderFeeder( other ), renderFeederToLock_( other.renderFeederToLock_ )
{
	// Nothing to do.
}

OpenSteer::Graphics::LockingRenderFeeder::~LockingRenderFeeder()
{
	// Nothing to do.
}



OpenSteer::Graphics::LockingRenderFeeder& 
OpenSteer::Graphics::LockingRenderFeeder::operator=( LockingRenderFeeder other )
{
	swap( other );
	return *this;
}

void 
OpenSteer::Graphics::LockingRenderFeeder::swap( LockingRenderFeeder& other )
{
	renderFeederToLock_.swap( other.renderFeederToLock_ );
}


void 
OpenSteer::Graphics::LockingRenderFeeder::setRenderFeederToLock( SharedPointer< RenderFeeder > const& renderFeederToLock )
{
	if ( 0 != renderFeederToLock ) {
		renderFeederToLock_ = renderFeederToLock;
	} else {
		renderFeederToLock_.reset( new NullRenderFeeder() );
	}
}



OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > 
OpenSteer::Graphics::LockingRenderFeeder::renderFeederToLock() const
{
	return renderFeederToLock_;
}



void 
OpenSteer::Graphics::LockingRenderFeeder::render( Matrix const& matrix, InstanceId const& instanceId )
{
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		renderFeederToLock_->render( matrix, instanceId );
	}
}



void 
OpenSteer::Graphics::LockingRenderFeeder::render( InstanceId const& instanceId )
{
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		renderFeederToLock_->render( instanceId );
	}	
}



void 
OpenSteer::Graphics::LockingRenderFeeder::render( Matrix const& matrix, GraphicsPrimitive const& graphicsPrimitive )
{
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		renderFeederToLock_->render( matrix, graphicsPrimitive );
	}	
}



void 
OpenSteer::Graphics::LockingRenderFeeder::render( GraphicsPrimitive const& graphicsPrimitive )
{
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		renderFeederToLock_->render( graphicsPrimitive );
	}	
}



bool
OpenSteer::Graphics::LockingRenderFeeder::addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& graphicsPrimitive, InstanceId& instanceId )
{
	bool return_value = false;
	
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		return_value = renderFeederToLock_->addToGraphicsPrimitiveLibrary( graphicsPrimitive, instanceId );
	}
	
	return return_value;
}



void 
OpenSteer::Graphics::LockingRenderFeeder::removeFromGraphicsPrimitiveLibrary( InstanceId const& instanceId )
{
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		renderFeederToLock_->removeFromGraphicsPrimitiveLibrary( instanceId );
	}	
}



bool 
OpenSteer::Graphics::LockingRenderFeeder::inGraphicsPrimitiveLibrary( InstanceId const& instanceId ) const
{
	bool return_value = false;
	
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
		{
			return_value = renderFeederToLock_->inGraphicsPrimitiveLibrary( instanceId );
		}	
		
		return return_value;
}



void 
OpenSteer::Graphics::LockingRenderFeeder::clearGraphicsPrimitiveLibrary()
{
	// bknafla: @todod Reread about the critical section naming scheme.
	#pragma omp critical
	{
		renderFeederToLock_->clearGraphicsPrimitiveLibrary();
	}	
}


