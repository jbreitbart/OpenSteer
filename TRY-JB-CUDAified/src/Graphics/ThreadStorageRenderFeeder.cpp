#include "OpenSteer/Graphics/ThreadStorageRenderFeeder.h"


// typedef ThreadStorage< SharedPointer< RenderFeeder > > ThreadStorageType;


OpenSteer::Graphics::ThreadStorageRenderFeeder::ThreadStorageRenderFeeder()
: threadStorage_( new ThreadStorageType( ThreadStorageType::default_size(), 
										 SharedPointer< RenderFeeder >( new NullRenderFeeder() ) ) )
{
	// Nothing to do.
}

OpenSteer::Graphics::ThreadStorageRenderFeeder::ThreadStorageRenderFeeder( SharedPointer< ThreadStorageType > const& _threadStorage )
: threadStorage_( _threadStorage )
{
	assert( threadStorage_ &&  "Smart pointer to thread storage points to 0. Behavior is undefined for further operations." );
}

OpenSteer::Graphics::ThreadStorageRenderFeeder::ThreadStorageRenderFeeder( ThreadStorageRenderFeeder const& other )
: threadStorage_( other.threadStorage_ )
{
	// Nothing to do.
}

OpenSteer::Graphics::ThreadStorageRenderFeeder::~ThreadStorageRenderFeeder()
{
	// Nothing to do.
}

OpenSteer::Graphics::ThreadStorageRenderFeeder& 
OpenSteer::Graphics::ThreadStorageRenderFeeder::operator=( ThreadStorageRenderFeeder other )
{
	swap( other );
	return *this;
}

void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::swap( ThreadStorageRenderFeeder& other )
{
	threadStorage_.swap( other.threadStorage_ );
}


void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::setThreadStorage(  SharedPointer< ThreadStorageType > const& _threadStorage )
{
	assert( _threadStorage &&
			"Smart pointer to thread storage points to 0. Behavior is undefined for further operations." );
	threadStorage_ = _threadStorage;
}



OpenSteer::SharedPointer< OpenSteer::Graphics::ThreadStorageRenderFeeder::ThreadStorageType > const& 
OpenSteer::Graphics::ThreadStorageRenderFeeder::threadStorage() const
{
	return threadStorage_;
}

void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::render( Matrix const& _matrix, 
														InstanceId const& _instanceId )
{
	threadStorage_->slot()->render( _matrix, _instanceId );
}



void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::render( InstanceId const& _instanceId )
{
	threadStorage_->slot()->render( _instanceId );
}



void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::render( Matrix const& _matrix, 
														GraphicsPrimitive const& _graphicsPrimitive )
{
	threadStorage_->slot()->render( _matrix, _graphicsPrimitive );
}


void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::render( GraphicsPrimitive const& _graphicsPrimitive )
{
	threadStorage_->slot()->render( _graphicsPrimitive );
}



bool 
OpenSteer::Graphics::ThreadStorageRenderFeeder::addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _graphicsPrimitive, 
																			   InstanceId& _instanceId )
{
	return threadStorage_->slot()->addToGraphicsPrimitiveLibrary( _graphicsPrimitive, _instanceId );
}


void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::removeFromGraphicsPrimitiveLibrary( InstanceId const& _instanceId )
{
	threadStorage_->slot()->removeFromGraphicsPrimitiveLibrary( _instanceId );
}


bool 
OpenSteer::Graphics::ThreadStorageRenderFeeder::inGraphicsPrimitiveLibrary( InstanceId const& _instanceId ) const
{
	return threadStorage_->slot()->inGraphicsPrimitiveLibrary( _instanceId );
}


void 
OpenSteer::Graphics::ThreadStorageRenderFeeder::clearGraphicsPrimitiveLibrary()
{
	threadStorage_->slot()->clearGraphicsPrimitiveLibrary();
}


