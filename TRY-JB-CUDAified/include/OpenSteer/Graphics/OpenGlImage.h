#ifndef OPENSTEER_GRAPHICS_OPENGLIMAGE_H
#define OPENSTEER_GRAPHICS_OPENGLIMAGE_H


// Include std::ostream, std::endl
#include <ostream>


// Include OpenGL types and enums.
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"

// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::SharedArray
#include "OpenSteer/SharedArray.h"

// Include OpenSteer::Color
#include "OpenSteer/Color.h"

// Include OPENSTEER_COMPILE_TIME_ASSERT
#include "OpenSteer/CompileTimeAssertion.h"


namespace OpenSteer {
    
    
    namespace Graphics {
        
        
        
        /**
         * @todo It would be interesting to examine a filter concept where a
         *       filter is applied when accessing (readig/setting) pixel 
         *       elements.
         *
         * @todo Change the order of height and width to resemble their usage
         *       when defining a multi-dimensional array or keep the x-axis, 
         *       y-axis order and then document the difference in usage of
         *       native arrays and @c OpenGlImage?
         */
        class OpenGlImage {
        public:
            typedef size_t size_type;
            typedef GLuint pixel_color_type;
            typedef GLubyte pixel_color_element_type;
            
            enum PixelColorElementIndex { R = 0, G = 1, B = 2, A = 3 };
            static size_type const pixelElementCount = 4;

            OPENSTEER_COMPILE_TIME_ASSERT( sizeof(pixel_color_type) == pixelElementCount * sizeof(pixel_color_element_type), 
                                           Four_pixel_color_element_types_must_have_the_same_size_as_one_pixel_color_type );
            
            /* One idea how to prevent the need for reinterpret_cast
            union Pixel {
                pixel_color_type color_;
                pixel_color_element_type colorElements_[ pixelElementCount ];
            };
            */
            
            
            
            OpenGlImage();
            OpenGlImage( size_type _width, size_type _height );
            OpenGlImage( size_type _width, size_type _height, pixel_color_type const& _color );
            OpenGlImage( size_type _width, 
                         size_type _height, 
                         pixel_color_element_type const& _r, 
                         pixel_color_element_type const& _g, 
                         pixel_color_element_type const& _b, 
                         pixel_color_element_type const& _a  );
            OpenGlImage( size_type _width, size_type _height, pixel_color_type const* _image );
            OpenGlImage( size_type _width, size_type _height, pixel_color_element_type const* _image );
            OpenGlImage( OpenGlImage const& _other );
            ~OpenGlImage();
            OpenGlImage& operator=( OpenGlImage _other );
            
            void swap( OpenGlImage& _other );
            
            size_type pixelCount() const {
                return width_ * height_;
            }
            
            size_type width() const {
                return width_;
            }
            
            size_type height() const {
                return height_;
            }
            
            pixel_color_type pixel( size_type _widthIndex, size_type _heightIndex ) const;
            pixel_color_element_type pixelElement( size_type _widthIndex, size_type _heightIndex, size_type _elementIndex ) const; 
            
            
            void setPixel( size_type _widthIndex, size_type _heightIndex, pixel_color_type const& _color );
            void setPixel( size_type _widthIndex, 
                           size_type _heightIndex,  
                           pixel_color_element_type const& _r, 
                           pixel_color_element_type const& _g, 
                           pixel_color_element_type const& _b, 
                           pixel_color_element_type const& _a );
            void setPixelElement( size_type _widthIndex, 
                                  size_type _heightIndex,  
                                  size_type _pixelElementIndex, 
                                  pixel_color_element_type const& _element );
            
            /*
            pixel_color_type const& operator()( size_type _widthIndex, size_type _heightIndex ) const;
            pixel_color_type& operator()( size_type _widthIndex, size_type _heightIndex );
            
            pixel_color_element_type const& operator()( size_type _widthIndex, size_type _heightIndex, PixelColorElementIndex _elementindex ) const;
            pixel_color_element_type& operator()( size_type _widthIndex, size_type _heightIndex, PixelColorElementIndex _elementindex );
            */
            
            void assign( size_type _width, size_type _height, pixel_color_type const* _image );
            void assign( size_type _width, size_type _height, pixel_color_element_type const* _image );
            
            void clear();
            void clear( pixel_color_type const& _color );
            void clear( pixel_color_element_type const& _r, 
                        pixel_color_element_type const& _g, 
                        pixel_color_element_type const& _b, 
                        pixel_color_element_type const& _a );
            
            pixel_color_element_type const* data() const;
            
            GLenum glPixelFormat() const;
            GLenum glPixelType() const;
            GLint glUnpackAlignment() const;
            GLenum glDimensionality() const;
            
        private:
            size_type width_;
            size_type height_;
            SharedArray< pixel_color_element_type > data_;
            
        }; // class OpenGlImage
        
        
       inline void swap( OpenGlImage& lhs, OpenGlImage& rhs ) {
           lhs.swap( rhs );
       }
        
       /**
        * Compares @a lhs and @a rhs for exact equality of all color elements
        * and returns @c true if the images are identical, @c false otherwise.
        */
       bool operator==( OpenGlImage const& lhs, OpenGlImage const& rhs );
       
       /**
        * Returns @c true if @a lhs and @a rhs are not equal, @c false if the
        * images are equal.
        */
       inline bool operator!=( OpenGlImage const& lhs, OpenGlImage const& rhs ) {
           return !( lhs == rhs );
       }
       
       
       
       OpenGlImage::pixel_color_element_type extractPixelColorElement( OpenGlImage::pixel_color_type const& _color, 
                                                                       OpenGlImage::size_type _index );
       
       OpenGlImage::pixel_color_type makePixelColor( OpenGlImage::pixel_color_element_type _r,
                                                     OpenGlImage::pixel_color_element_type _g,
                                                     OpenGlImage::pixel_color_element_type _b,
                                                     OpenGlImage::pixel_color_element_type _a );
       
       OpenGlImage::pixel_color_type makePixelColor( float _r,
                                                     float _g,
                                                     float _b,
                                                     float _a );
        
       OpenGlImage::pixel_color_type  makePixelColor( Color const& _color );
       
       OpenGlImage::pixel_color_element_type makePixelColorElement( float _value ); 
        
       /*
       void setPixelColorElement( OpenGlImage::pixel_color_type& _pixel,
                                  OpenGlImage::PixelColorElementIndex _elementIndex,
                                  OpenGlImage::pixel_color_element_type const& _value );
       
       void setPixelColorElement( OpenGlImage::pixel_color_type& _pixel,
                                  OpenGlImage::PixelColorElementIndex _elementIndex,
                                  float _value );
       */
       
       
       
       template< typename CharT, class Traits >
           std::basic_ostream< CharT, Traits >&
           operator<<( std::basic_ostream< CharT, Traits >& ostr, OpenGlImage const& image ) {
               
               // @todo Add iterators to @c OpenGlImage and then rewrite this
               //       with the @c std::for_each or something like it.
               for ( size_t height = 0; height != image.height(); ++height ) {
                   ostr << "{ ";
                   for ( size_t width = 0; width != image.width(); ++width ) {
                       ostr << "{ ";
                       for ( size_t element = 0; element != OpenGlImage::pixelElementCount; ++element ) {
                           ostr << static_cast<OpenGlImage::pixel_color_type>( image.pixelElement( width, height, element ) ) << " ";
                           
                       }
                       ostr << " }, ";
                   }
                   ostr << " }, " << std::endl;
               }
               ostr << " }";
               
               
               return ostr;
           }
       
       
       
       
    } // namespace Graphics
    
    
} // namespace OpenSteer




#endif // OPENSTEER_GRAPHICS_OPENGLIMAGE_H

