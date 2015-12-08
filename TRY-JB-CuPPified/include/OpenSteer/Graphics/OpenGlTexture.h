#ifndef OPENSTEER_GRAPHICS_OPENGLTEXTURE_H
#define OPENSTEER_GRAPHICS_OPENGLTEXTURE_H


// Include OpenGL, GLU, and Glut headers
#include "OpenSteer/Graphics/OpenGlHeaderWrapper.h"

// Include OpenSteer::Graphics::OpenGlImage
#include "OpenSteer/Graphics/OpenGlImage.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"



namespace OpenSteer {
    
    
    namespace Graphics {
        
        

        
        
        class OpenGlTexture {
        public:
            
            enum Wrapping { CLAMP = GL_CLAMP, 
                CLAMP_TO_EDGE = GL_CLAMP_TO_EDGE, 
                REPEAT = GL_REPEAT };
            
            enum MagFilter { MAG_NEAREST = GL_NEAREST,
                MAG_LINEAR = GL_LINEAR };
            
            enum MinFilter { MIN_NEAREST = GL_NEAREST,
                MIN_LINEAR = GL_LINEAR };
            
            
            OpenGlTexture();
            OpenGlTexture( SharedPointer< OpenGlImage > const& _image );
            OpenGlTexture( OpenGlTexture const& _other );
            
            ~OpenGlTexture();
            OpenGlTexture& operator=( OpenGlTexture _other );
            
            void swap( OpenGlTexture& _other );
            
            SharedPointer< OpenGlImage > image() const;
            void setImage( SharedPointer< OpenGlImage > const& _image );
            
            Wrapping wrapS() const;
            void setWrapS( Wrapping _wrapping );
            
            Wrapping wrapT() const;
            void setWrapT( Wrapping _wrapping );
            
            MagFilter magnificationFilter() const;
            void setMagnificationFilter( MagFilter _filter );
            
            MinFilter minificationFilter() const;
            void setMinificationFilter( MinFilter _filter );
            
            GLint border() const;
            bool borderEnabled() const;
            void enableBorder();
            void disableBorder();
            
            Color const& borderColor() const;
            void setBorderColor( Color const& _color );
            
            
            static float maxPriority();
            static float minPriority();
            
            float priority() const;
            void setPriority( float _priority );
            
            
        private:
            
            SharedPointer< OpenGlImage > image_;
            Color borderColor_;
            float priority_;
            Wrapping wrapS_;
            Wrapping wrapT_;
            MagFilter magnificationFilter_;
            MinFilter minificationFilter_;
            GLint border_;
        };
        
        
        /**
         * Swaps the content of @a lhs and @a rhs.
         */
        inline void swap( OpenGlTexture& lhs, OpenGlTexture& rhs ) {
            lhs.swap( rhs );
        }
        
        
        
        
    } // namespace Graphics
    
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_OPENGLTEXTURE_H
