#ifndef OPENSTEER_GRAPHICS_GRAPHICSPRIMITIVETRANSLATOR_H
#define OPENSTEER_GRAPHICS_GRAPHICSPRIMITIVETRANSLATOR_H

/*
class GraphicsPrimitiveTranslator {
public:
    virtual ~Translator() {}
    virtual Translator* clone() const = 0;
    virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive ) = 0;
    virtual void operator()( Matrix const& _transformation, GraphicsPrimitive const& _graphicsPrimitive ) = 0;
    
}; // class GraphicsPrimitiveTranslator
*/


// Include OpenSteer::Graphics::GraphicsPrimitiveTranslator
#include "OpenSteer/Graphics/OpenGlRenderService.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Graphics::LineGraphicsPrimitive, OpenSteer::Graphics::Vehicle2dGraphicsPrimitive, OpenSteer::Graphics::CircleGraphicsPrimitive, ...
#include "OpenSteer/Graphics/GraphicsPrimitives.h"

// Include OpenSteer::Graphics::createLine, OpenSteer::Graphics::createVehicleTriangle, OpenSteer::Graphics::createCircle, OpenSteer::Graphics::createDisc, OpenSteer::Graphics::createFloor
#include "OpenSteer/Graphics/OpenGlUtilities.h"

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"


namespace OpenSteer {
    
    
    namespace Graphics {
        
        // Forward declaration.
        // struct OpenGlRenderMesh;

        
        
        
        class LineOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            virtual ~LineOpenGlGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
             * If @a _primitive is not of type @c LineGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
             * If @a _primitive is not of type @c LineGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
             * If @a _primitive is not of type @c LineGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            
        }; // class LineOpenGlGraphicsPrimitiveToRendererTranslator
        
        
        
        
        
        
        
        
        // class LineOpenGlGraphicsPrimitiveTranslator : public OpenGlGraphicsPrimitiveTranslator {
        // public:
        //     
        //     virtual ~LineOpenGlGraphicsPrimitiveTranslator();
        //     
        //     // virtual LineOpenGlGraphicsPrimitiveTranslator* clone() const;
        // 
        //     /**
        //      * If @a _graphicsPrimitive is not of type @c LineGraphicsPrimitive 
        //      * a @c std::bad_cast is thrown.
        //      */
        //     virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     virtual void operator()( LineGraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        // }; // class OpenGlLineGraphicsPrimitiveTranslator
        
        
        
        
        class Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            virtual ~Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
             * If @a _primitive is not of type @c Vehicle2dGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
             * If @a _primitive is not of type @c Vehicle2dGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
             * If @a _primitive is not of type @c Vehicle2dGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            
        }; // class Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator
        
        
        
        
		class Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
		public:
            virtual ~Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
			 * If @a _primitive is not of type @c Vehicle2dGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
				* If @a _primitive is not of type @c Vehicle2dGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
				* If @a _primitive is not of type @c Vehicle2dGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            
        }; // class Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator
		
		
        
        
        
        // class Vehicle2dOpenGlGraphicsPrimitiveTranslator : public OpenGlGraphicsPrimitiveTranslator {
        // public:
        //     virtual ~Vehicle2dOpenGlGraphicsPrimitiveTranslator();
        //     
        //     // virtual Vehicle2dOpenGlGraphicsPrimitiveTranslator* clone() const;
        //     
        //     /**
        //      * If @a _graphicsPrimitive is not of type @c Vehicle2dGraphicsPrimitive 
        //      * a @c std::bad_cast is thrown.
        //      */
        //     virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     virtual void operator()( Vehicle2dGraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        // }; // class Vehicle2dOpenGlGraphicsPrimitiveTranslator
        
        
        
        
        template< size_t LineCount >
        class TrailLinesOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            virtual ~TrailLinesOpenGlGraphicsPrimitiveToRendererTranslator() {
                // Nothing to do.
            }
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const
            {
                return 0 != dynamic_cast< TrailLinesGraphicsPrimitive< LineCount > const* >( _primitive );
            }
            
            
            /**
             * If @a _primitive is not of type @c TrailLinesGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, 
                                       OpenGlRenderer& _renderer, 
                                       InstanceContainer& _instances ) const 
            {
            
                assert( translates( &_primitive) && "Translator called for wrong graphics primitive." );
                TrailLinesGraphicsPrimitive< LineCount > const& primitive = dynamic_cast< TrailLinesGraphicsPrimitive< LineCount > const& >( _primitive );
                
                InstanceId id = 0;
                bool const added = _renderer.addToRenderMeshLibrary( createTrail( primitive.trail(), 
                                                                                  primitive.material(),
                                                                                  primitive.tickMaterial() ) ,
                                                                     id );
                if ( added ) {
                    _instances.reserve( 1 );
                    _instances.push_back( id );
                }
                
                
                return added;
            }
            
            /**
                * If @a _primitive is not of type @c TrailLinesGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, 
                                    OpenGlRenderer& _renderer ) const {
            
                assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
                                    
                TrailLinesGraphicsPrimitive< LineCount > const& primitive = dynamic_cast< TrailLinesGraphicsPrimitive< LineCount > const& >( _primitive );                        
                                        
                _renderer.addToRender( createTrail( primitive.trail(), 
                                                    primitive.material(),
                                                    primitive.tickMaterial() ) );
            }

            /**
             * If @a _primitive is not of type @c TrailLinesGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const {
                
                assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
                
                TrailLinesGraphicsPrimitive< LineCount > const& primitive = dynamic_cast< TrailLinesGraphicsPrimitive< LineCount > const& >( _primitive );                        
                
                _renderer.addToRender( _transformation,
                                       createTrail( primitive.trail(), 
                                                    primitive.material(),
                                                    primitive.tickMaterial() ) );
            }
            
            
        }; // class TrailLinesOpenGlGraphicsPrimitiveToRendererTranslator
        
        
        
        
        
        
        
        /**
         * @todo Rewrite TrailLinesGraphicsPrimitive not to be template based to
         *       remove the template keyword here, too.
         */
        // template< size_t LineCount >
        // class TrailLinesOpenGlGraphicsPrimitiveTranslator : public OpenGlGraphicsPrimitiveTranslator {
        // public:
        //     virtual ~TrailLinesOpenGlGraphicsPrimitiveTranslator() {
        //         // Nothing to do.
        //     }
            
        //     virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore ) {
        //         operator()( dynamic_cast< TrailLinesGraphicsPrimitive< LineCount > const& >( _graphicsPrimitive ),
        //                     _meshStore );
        //     }
            
            
        //     virtual void operator()( TrailLinesGraphicsPrimitive< LineCount > const& _graphicsPrimitive, MeshContainer& _meshStore ) {
        //         _meshStore.push_back( createTrail( _graphicsPrimitive.trail(), 
        //               _graphicsPrimitive.material(),
        //               _graphicsPrimitive.tickMaterial() )  );
        //     }
        //     
        // }; // class TrailLinesOpenGlGraphicsPrimitiveTranslator
        
        

    class CircleOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
    public:
        virtual ~CircleOpenGlGraphicsPrimitiveToRendererTranslator();
    
        virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
        /**
         * If @a _primitive is not of type @c CircleGraphicsPrimitive 
         * a @c std::bad_cast is thrown.
         */
        virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
    
        /**
         * If @a _primitive is not of type @c CircleGraphicsPrimitive 
         * a @c std::bad_cast is thrown.
         */
        virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
        
        /**
         * If @a _primitive is not of type @c CircleGraphicsPrimitive 
         * a @c std::bad_cast is thrown.
         */
        virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
    
    
    }; // class CircleOpenGlGraphicsPrimitiveToRendererTranslator





        // class CircleOpenGlGraphicsPrimitiveTranslator : public OpenGlGraphicsPrimitiveTranslator {
        // public:
        //     
        //     virtual ~CircleOpenGlGraphicsPrimitiveTranslator();
        //     
        //     virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     virtual void operator()( CircleGraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     
        // }; // CircleOpenGlGraphicsPrimitiveTranslator
        
        


        class DiscOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            virtual ~DiscOpenGlGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
             * If @a _primitive is not of type @c DiscGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
             * If @a _primitive is not of type @c DiscGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
             * If @a _primitive is not of type @c DiscGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            
        }; // class DiscOpenGlGraphicsPrimitiveToRendererTranslator





        
        // class DiscOpenGlGraphicsPrimitiveTranslator : public OpenGlGraphicsPrimitiveTranslator {
        // public:
        //     
        //     virtual ~DiscOpenGlGraphicsPrimitiveTranslator();
        //     
        //     virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     virtual void operator()( DiscGraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     
        // }; // DiscOpenGlGraphicsPrimitiveTranslator
        
        

        class FloorOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            
            FloorOpenGlGraphicsPrimitiveToRendererTranslator();
            FloorOpenGlGraphicsPrimitiveToRendererTranslator( OpenGlRenderer::TextureId const& _id );
            virtual ~FloorOpenGlGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
             * If @a _primitive is not of type @c FloorGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
             * If @a _primitive is not of type @c FloorGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
             * If @a _primitive is not of type @c FloorGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
        
        private:
                
            OpenGlRenderer::TextureId textureId_;
            
        }; // class FloorOpenGlGraphicsPrimitiveToRendererTranslator





        
        // class FloorOpenGlGraphicsPrimitiveTranslator : public OpenGlGraphicsPrimitiveTranslator {
        // public:
        //     FloorOpenGlGraphicsPrimitiveTranslator();
        //     FloorOpenGlGraphicsPrimitiveTranslator( OpenGlRenderer::TextureId const& _id );
        //     
        //     virtual ~FloorOpenGlGraphicsPrimitiveTranslator();
        //     
        //     virtual void operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        //     
        //     virtual void operator()( FloorGraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore );
        // private:
        //     OpenGlRenderer::TextureId id_;
        //     
        // }; // FloorOpenGlGraphicsPrimitiveTranslator
        
        
        
        
        
        
        
        class TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            virtual ~TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
             * If @a _primitive is not of type @c TextAt3dLocationGraphicsPrimitive
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
             * If @a _primitive is not of type @c TextAt3dLocationGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
             * If @a _primitive is not of type @c TextAt3dLocationGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            
        }; // class TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator
        
        
        class TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator : public OpenGlGraphicsPrimitiveToRendererTranslator {
        public:
            virtual ~TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator();
            
            virtual bool translates( GraphicsPrimitive const* _primitive ) const;            
            /**
             * If @a _primitive is not of type @c TextAt2dLocationGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual bool addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const;
            
            /**
             * If @a _primitive is not of type @c TextAt2dLocationGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             */
            virtual void translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            /**
             * If @a _primitive is not of type @c TextAt2dLocationGraphicsPrimitive 
             * a @c std::bad_cast is thrown.
             * Throws away the matrix because it just positions text relative to the sceen.
             */
            virtual void translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const;
            
            
        }; // class TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator
        
        
    } // namespace Graphics
    
} // namespace OpenSteer



#endif // OPENSTEER_GRAPHICS_GRAPHICSPRIMITIVETRANSLATOR_H
