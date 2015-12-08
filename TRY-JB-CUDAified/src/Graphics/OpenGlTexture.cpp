#include "OpenSteer/Graphics/OpenGlTexture.h"

// Include std::swap
#include <algorithm>




namespace {
    
    float const minTexturePriority = 0.0f;
    float const maxTexturePriority = 1.0f;
    
    
} // anonymous namespace




OpenSteer::Graphics::OpenGlTexture::OpenGlTexture() 
    : image_(), borderColor_( 0.0f, 0.0f, 0.0f, 0.0f ), priority_( minTexturePriority ), wrapS_( CLAMP ), wrapT_( CLAMP ), magnificationFilter_( MAG_NEAREST ), minificationFilter_( MIN_NEAREST ), border_( 0 )
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlTexture::OpenGlTexture( SharedPointer< OpenGlImage > const& _image )
: image_( _image ), borderColor_( 0.0f, 0.0f, 0.0f, 0.0f ), priority_( minTexturePriority ), wrapS_( CLAMP ), wrapT_( CLAMP ), magnificationFilter_( MAG_NEAREST ), minificationFilter_( MIN_NEAREST ), border_( 0 )
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlTexture::OpenGlTexture( OpenGlTexture const& _other ) 
: image_( _other.image_), borderColor_(_other.borderColor_ ), priority_( _other.priority_ ), wrapS_( _other.wrapS_ ), wrapT_( _other.wrapT_ ), magnificationFilter_( _other.magnificationFilter_ ), minificationFilter_( _other.minificationFilter_ ), border_( 0 )
{
    // Nothing to do.
}




OpenSteer::Graphics::OpenGlTexture::~OpenGlTexture() 
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlTexture& 
OpenSteer::Graphics::OpenGlTexture::operator=( OpenGlTexture _other ) 
{
    swap( _other );
    return *this;
}




void 
OpenSteer::Graphics::OpenGlTexture::swap( OpenGlTexture& _other ) 
{
    image_.swap( _other.image_ );
    borderColor_.swap( _other.borderColor_ );
    std::swap( priority_, _other.priority_ );
    std::swap( wrapS_, _other.wrapS_ );
    std::swap( wrapT_, _other.wrapT_ );
    std::swap( magnificationFilter_, _other.magnificationFilter_ );
    std::swap( minificationFilter_, _other.minificationFilter_ );
    std::swap( border_, _other.border_ );
}




OpenSteer::SharedPointer< OpenSteer::Graphics::OpenGlImage > 
OpenSteer::Graphics::OpenGlTexture::image() const 
{
    return image_;
}



void 
OpenSteer::Graphics::OpenGlTexture::setImage( SharedPointer< OpenGlImage > const& _image ) 
{
    image_ = _image;
}




OpenSteer::Graphics::OpenGlTexture::Wrapping 
OpenSteer::Graphics::OpenGlTexture::wrapS() const 
{
    return wrapS_;
}



void 
OpenSteer::Graphics::OpenGlTexture::setWrapS( Wrapping _wrapping ) 
{
    wrapS_ = _wrapping;
}




OpenSteer::Graphics::OpenGlTexture::Wrapping 
OpenSteer::Graphics::OpenGlTexture::wrapT() const 
{
    return wrapT_;
}



void 
OpenSteer::Graphics::OpenGlTexture::setWrapT( Wrapping _wrapping ) 
{
    wrapT_ = _wrapping;
}




OpenSteer::Graphics::OpenGlTexture::MagFilter 
OpenSteer::Graphics::OpenGlTexture::magnificationFilter() const 
{
    return magnificationFilter_;
}



void 
OpenSteer::Graphics::OpenGlTexture::setMagnificationFilter( MagFilter _filter ) 
{
    magnificationFilter_ = _filter;
}




OpenSteer::Graphics::OpenGlTexture::MinFilter 
OpenSteer::Graphics::OpenGlTexture::minificationFilter() const 
{
    return minificationFilter_;
}



void 
OpenSteer::Graphics::OpenGlTexture::setMinificationFilter( MinFilter _filter ) 
{
    minificationFilter_ = _filter;
}



GLint 
OpenSteer::Graphics::OpenGlTexture::border() const
{
    return border_;
}



bool 
OpenSteer::Graphics::OpenGlTexture::borderEnabled() const
{
    return ( 1 == border_ );
}



void 
OpenSteer::Graphics::OpenGlTexture::enableBorder()
{
    border_ = 1;
}



void 
OpenSteer::Graphics::OpenGlTexture::disableBorder() 
{
    border_ = 0;
}



OpenSteer::Color const& 
OpenSteer::Graphics::OpenGlTexture::borderColor() const 
{
    return borderColor_;
}



void 
OpenSteer::Graphics::OpenGlTexture::setBorderColor( Color const& _color ) 
{
    borderColor_ = _color;
}


float
OpenSteer::Graphics::OpenGlTexture::maxPriority()
{
    return maxTexturePriority;
}



float
OpenSteer::Graphics::OpenGlTexture::minPriority() 
{
    return minTexturePriority;
}



float 
OpenSteer::Graphics::OpenGlTexture::priority() const 
{
    return priority_;
}





void 
OpenSteer::Graphics::OpenGlTexture::setPriority( float _priority ) 
{
    assert( ( ( 0.0f <= _priority ) && ( 1.0f >= _priority ) ) && "_priority is out of range." );
    priority_ = _priority;
}






