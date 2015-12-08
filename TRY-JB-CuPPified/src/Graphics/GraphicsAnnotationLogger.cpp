#include "OpenSteer/Graphics/GraphicsAnnotationLogger.h"

OpenSteer::Graphics::GraphicsAnnotationLogger::GraphicsAnnotationLogger()
    : nullRenderFeeder_(), renderFeeder_( &nullRenderFeeder_ ) 
{ 
    // Nothing to do.
}


OpenSteer::Graphics::GraphicsAnnotationLogger::GraphicsAnnotationLogger( RenderFeeder* _renderFeeder )
    : nullRenderFeeder_(), renderFeeder_( _renderFeeder ) 
{ 
    if ( 0 == _renderFeeder ) {
        renderFeeder_ = &nullRenderFeeder_;
    }
}



void 
OpenSteer::Graphics::GraphicsAnnotationLogger::setRenderFeeder( RenderFeeder* _renderFeeder )
{
    if ( 0 != _renderFeeder ) {
        renderFeeder_ = _renderFeeder;
    } else {
        renderFeeder_ = &nullRenderFeeder_;
    }
}
