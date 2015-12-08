#ifndef OPENSTEER_GRAPHICS_OPENGLRENDERSERVICE_H
#define OPENSTEER_GRAPHICS_OPENGLRENDERSERVICE_H

// Include OpenSteer::Graphics::RenderService, OpenSteer::Graphics::Renderer, OpenSteer::Graphics::Translator
#include "OpenSteer/Graphics/RenderService.h"


// Include std::vector
#include <vector>

// Include std::auto_ptr
#include <memory> 


// Include OpenSteer::size_t
#include "OpenSteer/StandardTypes.h"

// Include OpenSteer::Graphics::RenderFeeder
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::Graphics::GraphicsPrimitivesTranslatorMapper
#include "OpenSteer/Graphics/GraphicsPrimitivesTranslatorMapper.h"

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::Graphics::GraphicsPrimitives
#include "OpenSteer/Graphics/GraphicsPrimitives.h"

// Include OpenSteer::Graphics::OpenGlRenderPrimitive
#include "OpenSteer/Graphics/OpenGlRenderMesh.h"

// Include OpenSteer::Graphics::OpenGlRenderer
#include "OpenSteer/Graphics/OpenGlRenderer.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Graphics::OpenGlImage
#include "OpenSteer/Graphics/OpenGlImage.h"



namespace OpenSteer {
    
    namespace Graphics {
        
        /**
         * Abstract base class for translators taking a @c GraphicsPrimitive and 
         * if it is of the right type translate it and add it to @a _renderer.
         */
        /*
        class OpenGlGraphicsPrimitiveTranslator {
        public:
            
            typedef std::vector< SharedPointer< OpenGlRenderMesh > > MeshContainer;
            
            virtual ~OpenGlGraphicsPrimitiveTranslator() {};
            
            // virtual OpenGlGraphicsPrimitiveTranslator* clone() const;
            virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore ) = 0;
        }; // class OpenGlGraphicsPrimitiveTranslator
       */ 
        
        /**
         * Base class for translators that translate a specific kind of
         * graphics primitive and hand it to a renderer.
         */
        class OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            typedef OpenGlRenderer::InstanceId InstanceId;
            typedef std::vector< InstanceId > InstanceContainer;
            virtual ~OpenGlGraphicsPrimitiveToRendererTranslator();
            
            /**
             * Returns @c true if it is able to translate @a _primitive to the 
             * renderer, @c false otherwise.
             */
            virtual bool translates( GraphicsPrimitive const* _primitive ) const = 0;
            /**
             * @todo Is there a way to use an output iterator instead of a 
             *       container to return instance ids?
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const = 0;
            
            /**
             * Translates @a _primitive and hands it over to @a _renderer.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const = 0;
            
            
            
            /**
             * If @a _primitive is not of type @c LineGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const = 0;
            
        }; // class OpenGlGraphicsPrimitiveToRendererTranslator
        
        
        /**
         * Does nothing. Intended to be used as a default translator.
         */
        class NullOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            
            virtual ~NullOpenGlGraphicsPrimitiveToRendererTranslator() { /* Nothing to do. */ };
            
            virtual bool translates( GraphicsPrimitive const* /* _primitive */ ) const { return false; }
            virtual bool addToLibrary( GraphicsPrimitive const& /* _primitive */, OpenGlRenderer& /* _renderer */, InstanceContainer& /* _instances */ ) const { return false; }
            virtual void translate( GraphicsPrimitive const& /* _primitive */, OpenGlRenderer& /* _renderer */ ) const { /* Nothing to do. */ }
            virtual void translate( Matrix const& /* _transformation */, GraphicsPrimitive const& /* _primitive */, OpenGlRenderer& /* _renderer */ ) const { /* Nothing to do. */ }
        }; // class NullOpenGlGraphicsPrimitiveToRendererTranslator
        
        
        
        
        /**
         * Proxy for a real translator that is intended to be used in containers.
         */
        class OpenGlGraphicsPrimitiveToRendererTranslatorProxy : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            OpenGlGraphicsPrimitiveToRendererTranslatorProxy() : translator_( new NullOpenGlGraphicsPrimitiveToRendererTranslator() ) { /* Nothing to do. */ }
            explicit OpenGlGraphicsPrimitiveToRendererTranslatorProxy( SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > const& _translator ) : translator_( _translator ) {  if ( ! translator_ ) { translator_.reset( new NullOpenGlGraphicsPrimitiveToRendererTranslator() ); }  }
            virtual ~OpenGlGraphicsPrimitiveToRendererTranslatorProxy() { /* Nothing to do. */ };
            
            SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > translator() const { return translator_; }
            void setTranslator( SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > _translator ) { if ( _translator ) { translator_ = _translator; } else { translator_.reset( new NullOpenGlGraphicsPrimitiveToRendererTranslator() ); } }
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const { return translator_->translates( _primitive ); }
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const { return translator_->addToLibrary( _primitive, _renderer, _instances ); }
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const { translator_->translate( _primitive, _renderer ); }
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const { return translator_->translate( _transformation, _primitive, _renderer ); }
        private:
            
            SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > translator_;
            
        }; // class OpenGlGraphicsPrimitiveToRendererTranslatorProxy
        
        
        
        
        /**
         * Render service for a simple OpenGL renderer and its associated 
         * infrastructure.
         *
         * @attention Currently an instance needs to be present as long as 
         * any @c OpenGlRenderFeeder is used.
         */
        class OpenGlRenderService : public RenderService {
        public:
            virtual ~OpenGlRenderService();
            
            virtual SharedPointer< Renderer > createRenderer() const;
            
            /**
             * @a _renderer must be of type @c OpenGlRenderer.
             *
             * If @a _renderer does hold a @0-pointer or doesn't contain a valid
             * @c OpenGlRenderer a shared pointer with a @c 0-value is returned.
             */
            virtual SharedPointer< RenderFeeder > createRenderFeeder( SharedPointer< Renderer > const& _renderer ) const;
            
            void insertTranslator( std::type_info const& _typeInfo, SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > const& _translator );
            void removeTranslator( std::type_info const& _typeInfo );
            bool containsTranslator(  std::type_info const& _typeInfo ) const;
            void clearTranslators();
            
        private:
            
            GraphicsPrimitivesTranslatorMapper< SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > > translatorLookup_;
            // GraphicsPrimitivesTranslatorMapper< OpenGlGraphicsPrimitiveToRendererTranslatorProxy > translatorLookup_;
        }; // class OpenGlRenderService
        
        

        
        
       
        
        
        /**
         * Add mapping between @c RenderFeeder::InstanceId and @c OpenGlRenderer::InstanceId.
         *
         * @todo Need to adapt the translator mapper and the render feeder to 
         *       the new possibilities to just add null translators 
         *       (using the translator proxy) if an unknown primitive is tried 
         *       to be rendered. 
         *
         * @todo Remove old and unused code from the old translator handling.
         */
        class OpenGlRenderFeeder : public RenderFeeder {
        public:

            OpenGlRenderFeeder( SharedPointer< OpenGlRenderer > const& _rendererToFeed, 
                                GraphicsPrimitivesTranslatorMapper< SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > > const& _mapper );
            virtual ~OpenGlRenderFeeder();
            
            virtual bool addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _primitive, InstanceId& _id );
            // virtual bool addToGraphicsPrimitiveLibrary( Matrix const& _transformation, GraphicsPrimitive const& _primitive, InstanceId& _id );
            virtual void removeFromGraphicsPrimitiveLibrary( InstanceId const& _id );
            virtual bool inGraphicsPrimitiveLibrary( InstanceId const& _id ) const;
            virtual void clearGraphicsPrimitiveLibrary();
            
            virtual void render( Matrix const& _transformation, InstanceId const& _instanceId );
            virtual void render( InstanceId const& _instanceId );
            virtual void render( Matrix const& _transformation, GraphicsPrimitive const& _primitive );
            virtual void render( GraphicsPrimitive const& _primitive );
            
        private:
            /**
             * Not implemented to prevent copying.
             */
            OpenGlRenderFeeder( OpenGlRenderFeeder const& );
            
            /**
             * Not implemented to prevent assignment.
             */
            OpenGlRenderFeeder& operator=( OpenGlRenderFeeder const& );
            
        private:
            typedef std::vector< OpenGlRenderer::InstanceId > RendererInstanceIdContainer;
            typedef std::map< InstanceId, RendererInstanceIdContainer > GraphicsPrimitiveIdToRenderInstanceIdsMapping;
            GraphicsPrimitiveIdToRenderInstanceIdsMapping graphicsPrimitiveIdToRenderInstanceIdsMapping_;
            SharedPointer< OpenGlRenderer > renderer_;
            GraphicsPrimitivesTranslatorMapper< SharedPointer< OpenGlGraphicsPrimitiveToRendererTranslator > > const& translatorLookup_;
            // @todo Get @c const back!
            //GraphicsPrimitivesTranslatorMapper< OpenGlGraphicsPrimitiveToRendererTranslatorProxy > /* const */& translatorLookup_;
            InstanceId nextInstanceId_;
        
        }; // class OpenGlRenderFeeder
        
        
        
        
        
    } // namespace Graphics
    
    
} // namespace OpenSteer


#endif // OPENSTEER_GRAPHICS_OPENGLRENDERSERVICE_H


