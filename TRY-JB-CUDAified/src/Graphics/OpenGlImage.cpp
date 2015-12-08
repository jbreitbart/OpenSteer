#include "OpenSteer/Graphics/OpenGlImage.h"


// Include std::numerical_limits
#include <limits>



// Include std::fill_n, std::copy, std::swap, std::equal
#include <algorithm>

// Include assert
#include <cassert>



// Include OpenSteer::SharedArray
#include "OpenSteer/SharedArray.h"

// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include KAPAGA_UNUSED_PARAMETER
#include "kapaga/unused_parameter.h"


namespace {
    
    
    typedef size_t size_type;
    
    /**
     * Returns the index of an element stored in a flat array which is 
     * referenced by @a _widthIndex and @a _heightIndex into the virtual two
     * dimensional array of dimensions @a _width and @a _height.
     *
     * A @a _stride of @c 1 means that all elements are tightly packed, @c 2 
     * means that between two elements there is a potentially unused space of 
     * one element.
     *
     * @a _offset must be lesser than @a _stride and references one of the 
     * elements at the given index that are inside the stride.
     */
    size_t flatIndex( size_t _widthIndex, size_t _heightIndex, size_t _imageWidth, size_t _imageHeight, size_t _stride = 1, size_t _offset = 0 ) {
		KAPAGA_UNUSED_PARAMETER( _imageHeight );
		
        assert( _widthIndex < _imageWidth && "_widthIndex is out of range." );
        assert( _heightIndex < _imageHeight && "_heightIndex is out of range." );
        assert( 1 <= _stride && "Elements can't be packed tighter than one behind the other, therefore _stride must be at least 1." );
        assert( _offset < _stride && "_offset is to great and references the successor of the given index." );
                
        return ( ( _imageWidth * _heightIndex + _widthIndex ) * _stride ) + _offset;
    }
    
    
    
    /**
     * Translates a value @a valueToChange from scale 
     * <code> minSourceScaleValue - maxSourceScaleValue </code> including both
     * values and translates it into the scale
     * <code> minTargetScaleValue - maxTargetScaleValue </code> including both
     * values.
     *
     * @attention @c maxSourceScaleValue and @c minSourceScaleValue mustn't be
     *            equal.
     */
    float changeScale( float valueToChange, 
                       float minSourceScaleValue, 
                       float maxSourceScaleValue, 
                       float minTargetScaleValue, 
                       float maxTargetScaleValue) {
        assert( maxSourceScaleValue > minSourceScaleValue && 
                "minSourceScaleValue greater than maxSourceScaleValue. Distorted scale." );
        assert( valueToChange >= minSourceScaleValue && 
                "valueToChange lesser than minSourceScaleValue. Value not in scale." );
        assert( valueToChange <= maxSourceScaleValue && 
                "valueToChange greater than maxSourceScaleValue. Value not in scale." );
        assert( minTargetScaleValue <= maxTargetScaleValue && 
                "minTargetScaleValue greater than maxTargetScaleValue. Distorted scale." );
        
        float const sourceScaleRange = maxSourceScaleValue - minSourceScaleValue ;
        float const normalizedSourceValue = ( valueToChange - minSourceScaleValue ) / sourceScaleRange;
        float const targetScaleRange = maxTargetScaleValue - minTargetScaleValue;
        
        return ( normalizedSourceValue * targetScaleRange ) + minTargetScaleValue;        
    }
    
} // anonymous namespace




OpenSteer::Graphics::OpenGlImage::OpenGlImage()
    : width_( 1 ), height_( 1 ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{
    assert( width_ > 0 && "Width must be greater than 0 for a valid image." );
    assert( height_ > 0 && "Height must be greater than 0 for a valid image." );
        
    data_[ R ] = std::numeric_limits< pixel_color_element_type >::max();
    data_[ G ] = std::numeric_limits< pixel_color_element_type >::max();
    data_[ B ] = std::numeric_limits< pixel_color_element_type >::max();
    data_[ A ] = std::numeric_limits< pixel_color_element_type >::max() ;
}



OpenSteer::Graphics::OpenGlImage::OpenGlImage( size_type _width, size_type _height )
    : width_( _width ), height_( _height ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{
    assert( width_ > 0 && "Width must be greater than 0 for a valid image." );
    assert( height_ > 0 && "Height must be greater than 0 for a valid image." );
        
    std::fill_n( data_.get(), pixelCount() * pixelElementCount, std::numeric_limits< pixel_color_element_type >::max() );    
}



OpenSteer::Graphics::OpenGlImage::OpenGlImage( size_type _width, size_type _height, pixel_color_type const& _color )
    : width_( _width ), height_( _height ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{
    assert( width_ > 0 && "Width must be greater than 0 for a valid image." );
    assert( height_ > 0 && "Height must be greater than 0 for a valid image." );
        
    std::fill_n( reinterpret_cast< pixel_color_type* >( data_.get() ), pixelCount(), _color );     
}



OpenSteer::Graphics::OpenGlImage::OpenGlImage( size_type _width, 
                                               size_type _height, 
                                               pixel_color_element_type const& _r, 
                                               pixel_color_element_type const& _g, 
                                               pixel_color_element_type const& _b, 
                                               pixel_color_element_type const& _a  )
    : width_( _width ), height_( _height ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{
    assert( width_ > 0 && "Width must be greater than 0 for a valid image." );
    assert( height_ > 0 && "Height must be greater than 0 for a valid image." );
        
    std::fill_n( reinterpret_cast< pixel_color_type* >( data_.get() ), pixelCount(), makePixelColor( _r, _g, _b, _a ) ); 
}



OpenSteer::Graphics::OpenGlImage::OpenGlImage( size_type _width, size_type _height, pixel_color_type const* _image )
    : width_( _width ), height_( _height ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{
    assert( width_ > 0 && "Width must be greater than 0 for a valid image." );
    assert( height_ > 0 && "Height must be greater than 0 for a valid image." );
    
    std::copy( &_image[ 0 ], &_image[ pixelCount() ], reinterpret_cast< pixel_color_type* >( data_.get() ) );
}



OpenSteer::Graphics::OpenGlImage::OpenGlImage( size_type _width, size_type _height, pixel_color_element_type const* _image )
    : width_( _width ), height_( _height ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{
    assert( width_ > 0 && "Width must be greater than 0 for a valid image." );
    assert( height_ > 0 && "Height must be greater than 0 for a valid image." );
    
    std::copy( &_image[ 0 ], &_image[ pixelCount() * pixelElementCount ], data_.get() );    
}



OpenSteer::Graphics::OpenGlImage::OpenGlImage( OpenGlImage const& _other )
    : width_( _other.width_ ), height_( _other.height_ ), data_( new pixel_color_element_type[ width_ * height_ * pixelElementCount ] )
{    
    std::copy( &_other.data_[ 0 ], &_other.data_[ pixelCount() * pixelElementCount ], data_.get() );
    
}



OpenSteer::Graphics::OpenGlImage::~OpenGlImage()
{
    // Nothing to do.
}



OpenSteer::Graphics::OpenGlImage& 
OpenSteer::Graphics::OpenGlImage::operator=( OpenGlImage _other )
{
    swap( _other );
    return *this;
}




void 
OpenSteer::Graphics::OpenGlImage::swap( OpenGlImage& _other )
{
    std::swap( width_, _other.width_ );
    std::swap( height_, _other.height_ );
    data_.swap( _other.data_ );
}



/* Implemented inline...
OpenSteer::Graphics::OpenGlImage::size_type 
OpenSteer::Graphics::OpenGlImage::pixelCount() const
{
    
    
}



OpenSteer::Graphics::OpenGlImage::size_type 
OpenSteer::Graphics::OpenGlImage::width() const
{
    
    
}



OpenSteer::Graphics::OpenGlImage::size_type 
OpenSteer::Graphics::OpenGlImage::height() const
{
    
    
}
*/




OpenSteer::Graphics::OpenGlImage::pixel_color_type 
OpenSteer::Graphics::OpenGlImage::pixel( size_type _widthIndex, size_type _heightIndex ) const
{    
    return reinterpret_cast< pixel_color_type* >( data_.get() )[ flatIndex( _widthIndex, _heightIndex, width(), height() ) ];
}



OpenSteer::Graphics::OpenGlImage::pixel_color_element_type 
OpenSteer::Graphics::OpenGlImage::pixelElement( size_type _widthIndex, size_type _heightIndex, size_type _elementIndex ) const
{
    assert( pixelElementCount > _elementIndex && "_elementIndex out of range.");
    
    return data_[ flatIndex( _widthIndex, _heightIndex, width(), height(), pixelElementCount, _elementIndex ) ];
}





void 
OpenSteer::Graphics::OpenGlImage::setPixel( size_type _widthIndex, size_type _heightIndex, pixel_color_type const& _color )
{
    reinterpret_cast< pixel_color_type* >( data_.get() )[ flatIndex( _widthIndex, _heightIndex, width(), height() ) ] = _color;
}



void 
OpenSteer::Graphics::OpenGlImage::setPixel( size_type _widthIndex, 
                                            size_type _heightIndex,  
                                            pixel_color_element_type const& _r, 
                                            pixel_color_element_type const& _g, 
                                            pixel_color_element_type const& _b, 
                                            pixel_color_element_type const& _a )
{
    size_type const index = flatIndex( _widthIndex, _heightIndex, width(), height(), pixelElementCount );
    data_[ index + R ] = _r;
    data_[ index + G ] = _g;
    data_[ index + B ] = _b;
    data_[ index + A ] = _a;
}



void 
OpenSteer::Graphics::OpenGlImage::setPixelElement( size_type _widthIndex, 
                                                   size_type _heightIndex,  
                                                   size_type _pixelElementIndex, 
                                                   pixel_color_element_type const& _element )
{
    assert( pixelElementCount > _pixelElementIndex && "_pixelElementIndex out of range.");
    
    size_type const index = flatIndex( _widthIndex, _heightIndex, width(), height(), pixelElementCount, _pixelElementIndex );
    data_[ index ] = _element;
}




/*
 pixel_color_type const& operator()( size_type _widthIndex, size_type _heightIndex ) const;
 pixel_color_type& operator()( size_type _widthIndex, size_type _heightIndex );
 
 pixel_color_element_type const& operator()( size_type _widthIndex, size_type _heightIndex, PixelColorElementIndex _elementindex ) const;
 pixel_color_element_type& operator()( size_type _widthIndex, size_type _heightIndex, PixelColorElementIndex _elementindex );
 */

void 
OpenSteer::Graphics::OpenGlImage::assign( size_type _width, size_type _height, pixel_color_type const* _image )
{
    assert( _width > 0 && "Width must be greater than 0 for a valid image." );
    assert( _height > 0 && "Height must be greater than 0 for a valid image." );
    assert( 0 != _image && "_image must point to data." );
    
    SharedArray< pixel_color_element_type > newImage( new pixel_color_element_type[ _width * _height * pixelElementCount ] );
    
    std::copy( &_image[ 0 ], &_image[ _width * _height ], reinterpret_cast< pixel_color_type* >( newImage.get() ) );
    data_ = newImage;
    width_ = _width;
    height_ = _height;
}



void 
OpenSteer::Graphics::OpenGlImage::assign( size_type _width, size_type _height, pixel_color_element_type const* _image )
{
    assert( _width > 0 && "Width must be greater than 0 for a valid image." );
    assert( _height > 0 && "Height must be greater than 0 for a valid image." );
    assert( 0 != _image && "_image must point to data." );
    
    SharedArray< pixel_color_element_type > newImage( new pixel_color_element_type[ _width * _height * pixelElementCount ] );
    
    std::copy( &_image[ 0 ], &_image[ _width * _height * pixelElementCount ], newImage.get() );
    data_ = newImage;
    width_ = _width;
    height_ = _height;    
}




void 
OpenSteer::Graphics::OpenGlImage::clear()
{
    clear( std::numeric_limits< pixel_color_element_type >::max(), 
           std::numeric_limits< pixel_color_element_type >::max(),
           std::numeric_limits< pixel_color_element_type >::max(),
           std::numeric_limits< pixel_color_element_type >::max() );
}



void 
OpenSteer::Graphics::OpenGlImage::clear( pixel_color_type const& _color )
{
    std::fill_n( reinterpret_cast< pixel_color_type* >( data_.get() ), pixelCount() , _color );
}



void 
OpenSteer::Graphics::OpenGlImage::clear( pixel_color_element_type const& _r, 
                                         pixel_color_element_type const& _g, 
                                         pixel_color_element_type const& _b, 
                                         pixel_color_element_type const& _a )
{
    clear( makePixelColor( _r, _g, _b, _a ) );
}




OpenSteer::Graphics::OpenGlImage::pixel_color_element_type const* 
OpenSteer::Graphics::OpenGlImage::data() const
{
    return data_.get();
}




GLenum 
OpenSteer::Graphics::OpenGlImage::glPixelFormat() const
{
    return GL_RGBA;
}



GLenum 
OpenSteer::Graphics::OpenGlImage::glPixelType() const
{
    return GL_UNSIGNED_BYTE;    
}



GLint 
OpenSteer::Graphics::OpenGlImage::glUnpackAlignment() const
{
    return 1;
}

GLenum 
OpenSteer::Graphics::OpenGlImage::glDimensionality() const
{
    // @todo Differentiate for height or width equal to @c 0 and returning
    //       @c GL_TEXTURE_1D?
    return GL_TEXTURE_2D;
}


bool 
OpenSteer::Graphics::operator==( OpenSteer::Graphics::OpenGlImage const& lhs, 
                                 OpenSteer::Graphics::OpenGlImage const& rhs )
{
    // @todo Add tests for the OpenGL attributes the moment an image could hold
    //       different pixel data than currently.
    if ( ( lhs.width() != rhs.width() ) ||
         ( lhs.height() != rhs.height() ) ) {
        
        return false;
    }
         
    // @todo This isn't really clean. Add iterators to the OpenGlImage class.
    return std::equal( lhs.data(), lhs.data() + OpenGlImage::pixelElementCount * lhs.pixelCount(), rhs.data() );
}






OpenSteer::Graphics::OpenGlImage::pixel_color_element_type 
OpenSteer::Graphics::extractPixelColorElement( OpenGlImage::pixel_color_type const& _color, 
                                               OpenGlImage::size_type _index )
{    
    assert( OpenGlImage::pixelElementCount > _index && "_index out of range.");
    
    // @todo Do this with bit-magic.
    OpenGlImage::pixel_color_element_type const* pixelElements = reinterpret_cast< OpenGlImage::pixel_color_element_type const* >( &_color );
    return pixelElements[ _index ];
}




OpenSteer::Graphics::OpenGlImage::pixel_color_type 
OpenSteer::Graphics::makePixelColor( OpenGlImage::pixel_color_element_type _r,
                                     OpenGlImage::pixel_color_element_type _g,
                                     OpenGlImage::pixel_color_element_type _b,
                                     OpenGlImage::pixel_color_element_type _a )
{
    // @todo Do this with bit-magic.
    OpenGlImage::pixel_color_type pixel;
    OpenGlImage::pixel_color_element_type* pixelElements = reinterpret_cast< OpenGlImage::pixel_color_element_type* >( &pixel );
    pixelElements[ OpenGlImage::R ] = _r;
    pixelElements[ OpenGlImage::G ] = _g;
    pixelElements[ OpenGlImage::B ] = _b;
    pixelElements[ OpenGlImage::A ] = _a; 
    
    return pixel;
}




OpenSteer::Graphics::OpenGlImage::pixel_color_type 
OpenSteer::Graphics::makePixelColor( float _r,
                                     float _g,
                                     float _b,
                                     float _a )
{
    return makePixelColor( makePixelColorElement( _r ),
                           makePixelColorElement( _g ),
                           makePixelColorElement( _b ),
                           makePixelColorElement( _a )  );
}




OpenSteer::Graphics::OpenGlImage::pixel_color_type  
OpenSteer::Graphics::makePixelColor( Color const& _color )
{
    return makePixelColor( _color.r(), _color.g(), _color.b(), _color.a() );
    
}




OpenSteer::Graphics::OpenGlImage::pixel_color_element_type 
OpenSteer::Graphics::makePixelColorElement( float _value )
{
    return static_cast< OpenGlImage::pixel_color_type >( changeScale( _value, 
                                                                      0.0f, 
                                                                      1.0f, 
                                                                      0.0f, 
                                                                      std::numeric_limits< OpenGlImage::pixel_color_element_type >::max() ) );
}


