#ifndef OPENSTEER_GRAPHICS_OPENGLIMAGE_H
#define OPENSTEER_GRAPHICS_OPENGLIMAGE_H

// Include std::distance
#include <iterator>

// Include std::copy
#include <algorithm>



// Include OpenGL types and enums.
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"

// Include Opensteer::Color
#include "OpenSteer/Color.h"

// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::SharedArray
#include "OpenSteer/SharedArray.h"


namespace OpenSteer {
    
    namespace Grapics {
        
        /**
        * Image for usage for OpenGL textures.
         *
         * Pixel data represents RGBA data and is stored in a single 
         * @c value_type value. To extract red, green, blue, or alpha value of
         * a pixel the pixel data type is accesses with helper functions.
         *
         * Internally every color component/element (red, green, blue, alpha) is
         * stored with eight bits. Therefore each value can range from @c 0 to
         * @c 255. To translate an element value to a @c float it has to be
         * matched into the scale from @c 0.0f to @c 1.0f. These are the values
         * used by OpenGL, see Woo, et al, OpenGL Programming Guide, 
         * Third Edition, Addison-Wesley 1999, pp.306--311.
         *
         * @todo Add functionality to place sub-images into an image or to 
         *       extract a sub-image.
         *
         * @todo Replace @c new with a settable memory allocator.
         *
         * @todo Add a compile-time assertion that @c value_type is four times
         *       the size of @c byte.
         *
         * @todo Change the image representation to use more appropriate data 
         *       types that don't need complex index and casting operations.
         *
         * @todo Remove member functions using @c float and offer translation
         *       functions from @float to the used pixel representation in 
         *       appropriate utility headers.
         */
        class OpenGlImage {
public:
            typedef size_t size_type;
            typedef GLuint value_type;
            typedef GLubyte byte;
            
            /**
                * Creates an empty image with width and height of @c 1 and of color
             * white.
             */
            OpenGlImage();
            
            /**
                * Creates an empty image with the dimensions @a _width and 
             * @a _height and all white pixels.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            OpenGlImage( size_type _width, size_type _height );
            
            /**
                * Creates an empty image with the dimensions @a _width and 
             * @a _height and all pixels in the given color.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            // OpenGlImage( size_type _width, size_type _height, Color const& _color );
            
            /**
                * Creates an empty image with the dimensions @a _width and 
             * @a _height and all pixels in the given color.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            OpenGlImage( size_type _width, size_type _height, value_type const& _color );
            
            /**
                * Creates an empty image with the dimensions @a _width and 
             * @a _height and all pixels in the given color.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            // OpenGlImage( size_type _width, size_type _height, float _r, float _g, float _b, float _a );
            
            /**
                * Creates an empty image with the dimensions @a _width and 
             * @a _height and all pixels in the given color.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            OpenGlImage( size_type _width, size_type _height, byte const& _r, byte const& _g, byte const& _b, byte const& _a );
            
            /**
                * Creates a new image with the dimensions @a _width and @a _height 
             * and copies the data from @a _first to @a _last 
             * (excluding @a _last) into it.
             *
             * @c InputIterator must be an iterator over a data type that is 
             * assignable to @c value_type - that represents a whole pixel.
             *
             * The number of values referenced by @a _first and @a _last must be
             * lesser or equal than <code>_width * _height</code>.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            template< typename InputIterator >
                OpenGlImage( size_type _width, size_type _height, InputIterator _first, InputIterator _last )  
                : width_( _width ), height_ ( _height ), image_( 0 ) 
            {
                    assign( width_, height_, _first, _last );
            }
            
            OpenGlImage( OpenGlImage const& _other );
            
            OpenGlImage& operator=( OpenGlImage _other );
            
            void swap( OpenGlImage& _other );
            
            /**
                * Replaces the old image with a new one with the dimensions 
             * @a _width and @a _height and copies the data from @a _first to
             * @a _last (excluding @a _last) into it.
             *
             * @c InputIterator must be an iterator over a data type that is 
             * assignable to @c value_type - that represents a whole pixel.
             *
             * The number of values referenced by @a _first and @a _last must be
             * lesser or equal than <code>_width * _height</code>.
             *
             * @attention If used as a texture source @a _width and @a _height 
             *            should be powers of two.
             */
            template< typename InputIterator >
                assign( size_type _width, size_type _height, InputIterator _first, InputIterator _last ) {
                    assert( ( ( _width * _height ) >= std::distance( _first, _last ) ) && "" );
                    width_ = _width;
                    height_ = _height;
                    // @todo Replace @c new by a settable memory allocator.
                    image_.reset( new value_type[ _width * _height ] );
                    std::copy( first, last, image_.get() );
                }
            
            /**
                * Returns the width of the image.
             */
            size_type width() const {
                return width_;
            }
            
            /**
                * Returns the height of the image.
             */
            size_type height() const {
                return height_;
            }
            
            size_type size() const {
                return width() * height();
            }
            
            
            size_type pixelCount() const {
                return size();
            }
            
            void clear();
            // void clear( Color const& _clearColor );
            void clear( value_type const& _clearColor );
            void clear( byte const& _r, byte const& _g, byte const& _b, byte const& _a );
            // void clear( float _r, float _g, float _b, float _a );
            
            
            byte const& pixelElementR( size_type _widthIndex, size_type _heightIndex ) const;
            byte const& pixelElementG( size_type _widthIndex, size_type _heightIndex ) const;
            byte const& pixelElementB( size_type _widthIndex, size_type _heightIndex ) const;
            byte const& pixelElementA( size_type _widthIndex, size_type _heightIndex ) const;
            
            // Color const& pixelColor( size_type _widthIndex, size_type _heightIndex ) const;
            value_type const& pixelValue( size_type _widthIndex, size_type _heightIndex ) const;
            
            // void setPixel( size_type _widthIndex, size_type _heightIndex, Color const& _color );
            void setPixel( size_type _widthIndex, size_type _heightIndex, value_type const& _color );
            void setPixel( size_type _widthIndex, size_type _heightIndex, byte const& _r, byte const& _g, byte const& _b, byte const& _a );
            void setPixelElementR( size_type _widthIndex, size_type _heightIndex, byte const& _elementColor );
            void setPixelElementG( size_type _widthIndex, size_type _heightIndex, byte const& _elementColor );
            void setPixelElementB( size_type _widthIndex, size_type _heightIndex, byte const& _elementColor );
            void setPixelElementA( size_type _widthIndex, size_type _heightIndex, byte const& _elementColor );
            // void setPixelElementR( size_type _widthIndex, size_type _heightIndex, float _elementColor );
            // void setPixelElementG( size_type _widthIndex, size_type _heightIndex, float _elementColor );
            // void setPixelElementB( size_type _widthIndex, size_type _heightIndex, float _elementColor );
            // void setPixelElementA( size_type _widthIndex, size_type _heightIndex, float _elementColor );
            
            /**
                * Retuns a pointer to the image data encoded with the type of the
             * pixel elements. This can be passed directly to functions like:
             * @c glTexImage2D.
             *
             * Don't call if @c width() or @c height is @c 0.
             */
            byte const* data() const;
            
            /**
                * Returns the format of the image data, for example @c GL_RGBA.
             */
            GLenum format() const;
            
            /**
                * Returns the data type of the color format elements, for example
             * each color element (RGBA) is stored in an unsigned byte
             * @c GL_UNSIGNED_BYTE.
             */
            GLenum type() const;
            
            /**
                * Returns a number indicating how the bytes are packed in memory
             * so OpenGL can unpack them to feed them to the graphics hardware.
             *
             * Parameter to be used with 
             * <code>glPixelStorei( GL_UNPACK_ALIGNMENT, image.unpackAlignment() ) </code>
             * 
             * A number of @c 1 indicates that always the next available byte
             * is used in memory. A number of @c 4 means that the data of each
             * image row start at a 4-byte boundary in memory. See Woo, et al, 
             * OpenGL Programming Guide, Third Edition, Addison-Wesley 1999,
             * pp.306--311.
             */
            GLint unpackAlignment() const;
            
private:
                
                size_type width_;
            size_type height_;
            SharedArray< value_type > image_; 
            
        }; // class OpenGlImage
        
        
        void swap( OpenGlImage& lhs, OpenGlImage& rhs ) {
            lhs.swap( rhs );
        }
        
        
        void pixelElementR( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex, float& _r );
        void pixelElementG( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex, float& _g );
        void pixelElementB( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex, float& _b );
        void pixelElementA( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex, float& _a );
        void pixel( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex, Color& _color );
        void pixel( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex, float& _r, float& _g, float& _b, float& _a );
        float pixelElementRToFloat( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex );
        float pixelElementGToFloat( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex );
        float pixelElementBToFloat( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex );
        float pixelElementAToFloat( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex );
        Color pixelToColor( OpenGlImage const& _image, size_t _widthIndex, size_t _heightIndex );
        
        
        void setPixel( OpenGlImage& _image, size_t _widthIndex, size_t _heightIndex, float _r, float _g, float _b, float _a );
        void setPixel( OpenGlImage& _image, size_t _widthIndex, size_t _heightIndex, Color const& _color );
        void setPixelElementR( OpenGlImage& _image, size_t _widthIndex, size_t _heightIndex, float _r );
        void setPixelElementG( OpenGlImage& _image, size_t _widthIndex, size_t _heightIndex, float _g );
        void setPixelElementB( OpenGlImage& _image, size_t _widthIndex, size_t _heightIndex, float _b );
        void setPixelElementA( OpenGlImage& _image, size_t _widthIndex, size_t _heightIndex, float _a );
        
    } // namespace Graphics
    
    
} // namespace OpenSteer



#endif // OPENSTEER_GRAPHICS_OPENGLIMAGE_H
