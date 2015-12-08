#include "OpenSteer/Graphics/OpenGlGraphicsPrimitivesTranslators.h"

// Include OpenSteer::Graphics::createLine, OpenSteer::Graphics::createVehicleTriangle, OpenSteer::Graphics::createCircle
#include "OpenSteer/Graphics/OpenGlRenderMesh.h"

// Include OpenSteer::Graphics::OpenGlRenderText
#include "OpenSteer/Graphics/OpenGlRenderText.h"

// Include OPENSTEER_UNUSED_PARAMETER
#include "OpenSteer/UnusedParameter.h"


OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveToRendererTranslator::~LineOpenGlGraphicsPrimitiveToRendererTranslator() 
{ 
    // Nothing to do.
}

bool 
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0 != dynamic_cast< LineGraphicsPrimitive const* >( _primitive );
}


bool 
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, 
                                                                                    OpenGlRenderer& _renderer, 
                                                                                    InstanceContainer& _instances ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    LineGraphicsPrimitive const& primitive = dynamic_cast< LineGraphicsPrimitive const& >( _primitive );
    
    InstanceId id = 0;
    bool const added = _renderer.addToRenderMeshLibrary(   createLine( primitive.lineBegin(), 
                                                                       primitive.lineEnd(), 
                                                                       primitive.material() ), id );
    
    if ( added ) {
        _instances.reserve( 1 );
        _instances.push_back( id );
    } 
    
    return added;
}

void 
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, 
                                                                                 OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    LineGraphicsPrimitive const& primitive = dynamic_cast< LineGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( createLine( primitive.lineBegin(), 
                                       primitive.lineEnd(), 
                                       primitive.material() ) );    
}



void 
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation,
                                                                                 GraphicsPrimitive const& _primitive, 
                                                                                 OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    LineGraphicsPrimitive const& primitive = dynamic_cast< LineGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( _transformation,
                           createLine( primitive.lineBegin(), 
                                       primitive.lineEnd(), 
                                       primitive.material() ) );    
}



/*
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveTranslator::~LineOpenGlGraphicsPrimitiveTranslator()
{
    // Nothing to do.
}
*/


/*
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveTranslator* 
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveTranslator::clone() const
{
    return new LineOpenGlGraphicsPrimitiveTranslator();
}
*/

/*
void
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveTranslator::operator()( GraphicsPrimitive const& _graphicsPrimitive,
                                                                        MeshContainer& _meshStore )
{
    return operator()( dynamic_cast< LineGraphicsPrimitive const& >( _graphicsPrimitive ), _meshStore );
}







void 
OpenSteer::Graphics::LineOpenGlGraphicsPrimitiveTranslator::operator()( LineGraphicsPrimitive const& _graphicsPrimitive,
                                                                        MeshContainer& _meshStore )
{
    _meshStore.push_back( createLine( _graphicsPrimitive.lineBegin(), 
                                      _graphicsPrimitive.lineEnd(), 
                                      _graphicsPrimitive.material() ) );
}
*/






OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator::~Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



bool 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0 != dynamic_cast< Vehicle2dGraphicsPrimitive const* >( _primitive );
}


bool 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, 
                                                                                         OpenGlRenderer& _renderer, 
                                                                                         InstanceContainer& _instances ) const
{
    assert( translates( &_primitive ) && "Translator called for a wrong graphics primitive." );
    
    Vehicle2dGraphicsPrimitive const& primitive = dynamic_cast< Vehicle2dGraphicsPrimitive const& >( _primitive );
    
    InstanceId triangleId = 0;    
    bool added = _renderer.addToRenderMeshLibrary( createVehicleTriangle( primitive.radius(), 
                                                                                  primitive.material() ),
                                                   triangleId );
    
    if ( ! added ) {
        std::cerr << "Unable to add vehicle triangle, part of vehicle graphics, to the renderer mesh library." << std::endl;
        return false;
    }
    
    InstanceId circleId = 0;
    added = _renderer.addToRenderMeshLibrary( createCircle( primitive.radius(),
                                                            primitive.circleSegmentCount(),
                                                            primitive.circleMaterial() ),
                                              circleId );
    
    if ( ! added ) {
        _renderer.removeFromRenderMeshLibrary( triangleId );
        std::cerr << "Unable to add vehicle circle, part of vehicle graphics, to the renderer mesh library." << std::endl;
        return false;
    }
    
    _instances.reserve( 2 );
    _instances.push_back( triangleId );
    _instances.push_back( circleId );
    
    return true;
}


void 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, 
                                                                                      OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for a wrong graphics primitive." );
    
    Vehicle2dGraphicsPrimitive const& primitive = dynamic_cast< Vehicle2dGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( createVehicleTriangle( primitive.radius(), 
                                                  primitive.material() ) );
    
    _renderer.addToRender( createCircle( primitive.radius(),
                                         primitive.circleSegmentCount(),
                                         primitive.circleMaterial() ) );
}



void 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation,
                                                                                      GraphicsPrimitive const& _primitive, 
                                                                                      OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for a wrong graphics primitive." );
    
    Vehicle2dGraphicsPrimitive const& primitive = dynamic_cast< Vehicle2dGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( _transformation,
                           createVehicleTriangle( primitive.radius(), 
                                                  primitive.material() ) );
    
    _renderer.addToRender( _transformation,
                           createCircle( primitive.radius(),
                                         primitive.circleSegmentCount(),
                                         primitive.circleMaterial() ) );
}







/**************************************************************************************************/ 

OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator::~Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



bool 
OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0 != dynamic_cast< Basic3dSphericalVehicleGraphicsPrimitive const* >( _primitive );
}


bool 
OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, 
                                                                                         OpenGlRenderer& _renderer, 
                                                                                         InstanceContainer& _instances ) const
{
    assert( translates( &_primitive ) && "Translator called for a wrong graphics primitive." );
    
    Basic3dSphericalVehicleGraphicsPrimitive const& primitive = dynamic_cast< Basic3dSphericalVehicleGraphicsPrimitive const& >( _primitive );
    
    InstanceId vehicleId = 0;    
    bool added = _renderer.addToRenderMeshLibrary( createBasic3dSphericalVehicle( primitive.length_center(),
																				  primitive.length(),
																				  primitive.width(),
																				  primitive.height(), 
																				  primitive.material(),
																				  primitive.material_variation_factor() ),
                                                   vehicleId );
    
    if ( ! added ) {
        std::cerr << "Unable to add basic 3d spherical vehicle to the renderer mesh library." << std::endl;
        return false;
    }
    
    
    _instances.reserve( 1 );
    _instances.push_back( vehicleId );
    
    return true;
}


void 
OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, 
                                                                                      OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for a wrong graphics primitive." );
    
    Basic3dSphericalVehicleGraphicsPrimitive const& primitive = dynamic_cast< Basic3dSphericalVehicleGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( createBasic3dSphericalVehicle( primitive.length_center(),
														  primitive.length(),
														  primitive.width(),
														  primitive.height(), 
														  primitive.material(),
														  primitive.material_variation_factor() ) );
}



void 
OpenSteer::Graphics::Basic3dSphericalVehicleGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation,
                                                                                      GraphicsPrimitive const& _primitive, 
                                                                                      OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for a wrong graphics primitive." );
    
    Basic3dSphericalVehicleGraphicsPrimitive const& primitive = dynamic_cast< Basic3dSphericalVehicleGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( _transformation,
                           createBasic3dSphericalVehicle( primitive.length_center(),
														  primitive.length(),
														  primitive.width(),
														  primitive.height(), 
														  primitive.material(),
														  primitive.material_variation_factor() ) );
	
}


/**************************************************************************************************/ 










/*
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveTranslator::~Vehicle2dOpenGlGraphicsPrimitiveTranslator()
{
    // Nothing to do.
}
*/


/*
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveTranslator* 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveTranslator::clone() const
{
    return new Vehicle2dOpenGlGraphicsPrimitiveTranslator();
}
*/

/*
void 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveTranslator::operator()( GraphicsPrimitive const& _graphicsPrimitive,
                                                                           MeshContainer& _meshStore )
{
    return operator()( dynamic_cast< Vehicle2dGraphicsPrimitive const& >( _graphicsPrimitive ), _meshStore );
}






void 
OpenSteer::Graphics::Vehicle2dOpenGlGraphicsPrimitiveTranslator::operator()( Vehicle2dGraphicsPrimitive const& _graphicsPrimitive,
                                                                           MeshContainer& _meshStore )
{    
    _meshStore.push_back( createVehicleTriangle( _graphicsPrimitive.radius(), 
                                                 _graphicsPrimitive.material() ) );
    
    _meshStore.push_back( createCircle( _graphicsPrimitive.radius(),
                                        _graphicsPrimitive.circleSegmentCount(),
                                        _graphicsPrimitive.circleMaterial() ) );
}
*/





OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveToRendererTranslator::~CircleOpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}

bool 
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0 != dynamic_cast< CircleGraphicsPrimitive const* >( _primitive );
}


bool 
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, 
                                                                                      OpenGlRenderer& _renderer, 
                                                                                      InstanceContainer& _instances ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    CircleGraphicsPrimitive const& circlePrimitive = dynamic_cast< CircleGraphicsPrimitive const& >( _primitive );
    
    InstanceId id = 0;
    bool const added = _renderer.addToRenderMeshLibrary( createCircle( circlePrimitive.radius(),
                                                                       circlePrimitive.segmentCount(),
                                                                       circlePrimitive.material() ),
                                                         id );
    
    if ( added ) {
        _instances.reserve( 1 );
        _instances.push_back( id );
    }

    return added;
}



void 
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, 
                                                                                   OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );

    CircleGraphicsPrimitive const& circlePrimitive = dynamic_cast< CircleGraphicsPrimitive const& >( _primitive );

    _renderer.addToRender( createCircle( circlePrimitive.radius(),
                                         circlePrimitive.segmentCount(),
                                         circlePrimitive.material() ) );
}


void 
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation,
                                                                                   GraphicsPrimitive const& _primitive, 
                                                                                   OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );

CircleGraphicsPrimitive const& circlePrimitive = dynamic_cast< CircleGraphicsPrimitive const& >( _primitive );

_renderer.addToRender( _transformation,
                       createCircle( circlePrimitive.radius(),
                                     circlePrimitive.segmentCount(),
                                     circlePrimitive.material() ) );
}






/*
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveTranslator::~CircleOpenGlGraphicsPrimitiveTranslator()
{
    // Nothing to do.
}



void 
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveTranslator::operator()( GraphicsPrimitive const& _graphicsPrimitive, 
                                                                          MeshContainer& _meshStore )
{
    return operator()( dynamic_cast< CircleGraphicsPrimitive const& >( _graphicsPrimitive ), _meshStore );
}



void 
OpenSteer::Graphics::CircleOpenGlGraphicsPrimitiveTranslator::operator()( CircleGraphicsPrimitive const& _graphicsPrimitive, 
                                                                          MeshContainer& _meshStore )
{
    _meshStore.push_back( createCircle( _graphicsPrimitive.radius(),
                                        _graphicsPrimitive.segmentCount(),
                                        _graphicsPrimitive.material() ) );
}
*/





OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveToRendererTranslator::~DiscOpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



bool 
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0 != dynamic_cast< DiscGraphicsPrimitive const* >( _primitive );
}



bool 
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, 
                                                                                    OpenGlRenderer& _renderer, 
                                                                                    InstanceContainer& _instances ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    DiscGraphicsPrimitive const& discPrimitive = dynamic_cast< DiscGraphicsPrimitive const& >( _primitive );
    
    InstanceId id = 0;
    
    bool const added = _renderer.addToRenderMeshLibrary( createDisc( discPrimitive.radius(),
                                                                     discPrimitive.segmentCount(),
                                                                     discPrimitive.material() ),
                                                         id );
    
    if ( added ) {
        _instances.reserve( 1 );
        _instances.push_back( id );
    }
    
    
    return added;
}



void 
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, 
                                                                                 OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    DiscGraphicsPrimitive const& discPrimitive = dynamic_cast< DiscGraphicsPrimitive const& >( _primitive ); 
    
    _renderer.addToRender( createDisc( discPrimitive.radius(),
                                       discPrimitive.segmentCount(),
                                       discPrimitive.material() ) );
}



void 
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation,
                                                                                 GraphicsPrimitive const& _primitive, 
                                                                                 OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    DiscGraphicsPrimitive const& discPrimitive = dynamic_cast< DiscGraphicsPrimitive const& >( _primitive ); 
    
    _renderer.addToRender( _transformation,
                           createDisc( discPrimitive.radius(),
                                       discPrimitive.segmentCount(),
                                       discPrimitive.material() ) );
}


/*
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveTranslator::~DiscOpenGlGraphicsPrimitiveTranslator()
{
    // Nothing to do.
}



void 
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveTranslator::operator()( GraphicsPrimitive const& _graphicsPrimitive, 
                                                                          MeshContainer& _meshStore )
{
    return operator()( dynamic_cast< DiscGraphicsPrimitive const& >( _graphicsPrimitive ), _meshStore );
}



void 
OpenSteer::Graphics::DiscOpenGlGraphicsPrimitiveTranslator::operator()( DiscGraphicsPrimitive const& _graphicsPrimitive, 
                                                                          MeshContainer& _meshStore )
{
    _meshStore.push_back( createDisc( _graphicsPrimitive.radius(),
                                      _graphicsPrimitive.segmentCount(),
                                      _graphicsPrimitive.material() ) );
}
*/






OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::FloorOpenGlGraphicsPrimitiveToRendererTranslator()
: textureId_( 0 )
{
    // Nothing to do.
}



OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::FloorOpenGlGraphicsPrimitiveToRendererTranslator( OpenGlRenderer::TextureId const& _id )
: textureId_( _id )
{
    // Nothing to do.
}



OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::~FloorOpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



bool 
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0!= dynamic_cast< FloorGraphicsPrimitive const* >( _primitive );
}



bool 
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, 
                                                                                     OpenGlRenderer& _renderer, 
                                                                                     InstanceContainer& _instances ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    FloorGraphicsPrimitive const& floorPrimitive = dynamic_cast< FloorGraphicsPrimitive const& >( _primitive );
    
    InstanceId floorId = 0;
    bool const added = _renderer.addToRenderMeshLibrary( createFloor( floorPrimitive.breadth(), 
                                                                      floorPrimitive.length(), 
                                                                      floorPrimitive.material(), 
                                                                      textureId_ ),
                                                         floorId );
    if ( added ) {
        _instances.reserve( 1 );
        _instances.push_back( floorId );
    }
    
    return added;
}



void 
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, 
                                                                                  OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    FloorGraphicsPrimitive const& floorPrimitive = dynamic_cast< FloorGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( createFloor( floorPrimitive.breadth(), 
                                        floorPrimitive.length(), 
                                        floorPrimitive.material(), 
                                        textureId_ ) );
}



void 
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation,
                                                                                  GraphicsPrimitive const& _primitive, 
                                                                                  OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    FloorGraphicsPrimitive const& floorPrimitive = dynamic_cast< FloorGraphicsPrimitive const& >( _primitive );
    
    _renderer.addToRender( _transformation,
                           createFloor( floorPrimitive.breadth(), 
                                        floorPrimitive.length(), 
                                        floorPrimitive.material(), 
                                        textureId_ ) );
}


/*
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveTranslator::FloorOpenGlGraphicsPrimitiveTranslator()
: id_( 0 ) 
{
    // Nothing to do.
}



OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveTranslator::FloorOpenGlGraphicsPrimitiveTranslator( OpenGlRenderer::TextureId const& _id )
: id_( _id )
{
    // Nothing to do.
}

OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveTranslator::~FloorOpenGlGraphicsPrimitiveTranslator()
{
    // Nothing to do.
}



void 
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveTranslator::operator()( GraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore )
{
    return operator()( dynamic_cast< FloorGraphicsPrimitive const& >( _graphicsPrimitive ), _meshStore );
}



void 
OpenSteer::Graphics::FloorOpenGlGraphicsPrimitiveTranslator::operator()( FloorGraphicsPrimitive const& _graphicsPrimitive, MeshContainer& _meshStore )
{
    _meshStore.push_back( createFloor( _graphicsPrimitive.breadth(), _graphicsPrimitive.length(), _graphicsPrimitive.material(), id_ ) );    
}
*/





OpenSteer::Graphics::TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator::~TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



bool 
OpenSteer::Graphics::TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0!= dynamic_cast< TextAt3dLocationGraphicsPrimitive const* >( _primitive );
}



bool 
OpenSteer::Graphics::TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const
{
	OPENSTEER_UNUSED_PARAMETER( _primitive );
	OPENSTEER_UNUSED_PARAMETER( _renderer );
	OPENSTEER_UNUSED_PARAMETER( _instances );
    // @todo Implement functionality.
    return false;
}




void 
OpenSteer::Graphics::TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const
{
    translate( identityMatrix, _primitive, _renderer );
}




void 
OpenSteer::Graphics::TextAt3dLocationOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    TextAt3dLocationGraphicsPrimitive const& textPrimitive = dynamic_cast< TextAt3dLocationGraphicsPrimitive const& >( _primitive );
    
    SharedPointer< OpenGlRenderText > renderPrimitive( new OpenGlRenderText( textPrimitive.text(), textPrimitive.material() ) );
    
    _renderer.addTextToRender( _transformation,
                               renderPrimitive );    
}








OpenSteer::Graphics::TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator::~TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator()
{
    // Nothing to do.
}



bool 
OpenSteer::Graphics::TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator::translates( GraphicsPrimitive const* _primitive ) const
{
    return 0!= dynamic_cast< TextAt2dLocationGraphicsPrimitive const* >( _primitive );
}



bool 
OpenSteer::Graphics::TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator::addToLibrary( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer, InstanceContainer& _instances ) const
{
	OPENSTEER_UNUSED_PARAMETER( _primitive );
	OPENSTEER_UNUSED_PARAMETER( _renderer );
	OPENSTEER_UNUSED_PARAMETER( _instances );
    // @todo Implement functionality.
    return false;
}




void 
OpenSteer::Graphics::TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator::translate( GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const
{
    assert( translates( &_primitive ) && "Translator called for wrong graphics primitive." );
    
    TextAt2dLocationGraphicsPrimitive const& textPrimitive = dynamic_cast< TextAt2dLocationGraphicsPrimitive const& >( _primitive );
    
    
    OpenGlRenderText2d::RelativePosition relativePosition = OpenGlRenderText2d::TOP_LEFT;
    switch ( textPrimitive.relativePosition() ) {
        case TextAt2dLocationGraphicsPrimitive::TOP_LEFT:
            relativePosition = OpenGlRenderText2d::TOP_LEFT;
            break;
        case TextAt2dLocationGraphicsPrimitive::BOTTOM_LEFT:
            relativePosition = OpenGlRenderText2d::BOTTOM_LEFT;
            break;
        case TextAt2dLocationGraphicsPrimitive::TOP_RIGHT:
            relativePosition = OpenGlRenderText2d::TOP_RIGHT;
            break;
        case TextAt2dLocationGraphicsPrimitive::BOTTOM_RIGHT:
            relativePosition = OpenGlRenderText2d::BOTTOM_RIGHT;
            break;
        default:
            relativePosition = OpenGlRenderText2d::TOP_LEFT;
    }
    
    
    SharedPointer< OpenGlRenderText2d > renderPrimitive( new OpenGlRenderText2d( textPrimitive.position(), relativePosition, textPrimitive.text(), textPrimitive.material() ) );
    
    _renderer.addText2dToRender( renderPrimitive ); 
}




void 
OpenSteer::Graphics::TextAt2dLocationOpenGlGraphicsPrimitiveToRendererTranslator::translate( Matrix const& _transformation, GraphicsPrimitive const& _primitive, OpenGlRenderer& _renderer ) const
{   
    OPENSTEER_UNUSED_PARAMETER( _transformation );
    // Throws away the matrix because it just positions text relative to the sceen.
    translate( _primitive, _renderer );
}







