#include "OpenSteer/Graphics/BatchingRenderFeeder.h"


// Include std::for_each
#include <algorithm>

// Include std::unary_function
#include <functional>



namespace {
    
    typedef std::pair< OpenSteer::Matrix, OpenSteer::Graphics::RenderFeeder::InstanceId > TransformedInstance;
    typedef std::pair< OpenSteer::Matrix, OpenSteer::SharedPointer< OpenSteer::Graphics::GraphicsPrimitive > > TransformedGraphicsPrimitive;
    
    
    
    /**
     * Clones @a _primitive and returns the clone in a shared pointer.
     */
    OpenSteer::SharedPointer< OpenSteer::Graphics::GraphicsPrimitive > clone( OpenSteer::Graphics::GraphicsPrimitive const& _primitive ) {
        return OpenSteer::SharedPointer< OpenSteer::Graphics::GraphicsPrimitive >( _primitive.clone() );
    }
    
    
    class FeedTransformedInstances : public std::unary_function< TransformedInstance, void > {
    public:
        explicit FeedTransformedInstances( OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > const& _feeder )
        : renderFeeder_( _feeder ) {}
        
        void operator() ( TransformedInstance const& _argument ) {
            renderFeeder_->render( _argument.first, _argument.second );
        }
        
    private:
        OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > renderFeeder_;
    
    }; // class FeedTransformedInstances
    
    
    
    class FeedInstances : public std::unary_function< OpenSteer::Graphics::RenderFeeder::InstanceId, void > {
    public:    
        explicit FeedInstances( OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > const& _feeder )
        : renderFeeder_( _feeder ) {}
        
        void operator() ( OpenSteer::Graphics::RenderFeeder::InstanceId const& _argument ) {
            renderFeeder_->render( _argument );
        }
        
    private:
        OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > renderFeeder_;    
        
    }; // class FeedInstances
    
    
    
    class FeedTransformedGraphicsPrimitives : public std::unary_function< TransformedGraphicsPrimitive, void > {
    public:
        explicit FeedTransformedGraphicsPrimitives( OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > const& _feeder )
        : renderFeeder_( _feeder ) {}
        
        void operator() ( TransformedGraphicsPrimitive const& _argument ) {
            renderFeeder_->render( _argument.first, *_argument.second );
        }
        
    private:
        OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > renderFeeder_;
    }; // class FeedTransformedGraphicsPrimitives
    
    
    
    class FeedGraphicsPrimitives : public std::unary_function< OpenSteer::SharedPointer< OpenSteer::Graphics::GraphicsPrimitive >, void > {
    public:    
        explicit FeedGraphicsPrimitives( OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > const& _feeder )
        : renderFeeder_( _feeder ) {}
        
        void operator() ( OpenSteer::SharedPointer< OpenSteer::Graphics::GraphicsPrimitive > const& _argument ) {
            renderFeeder_->render( *_argument );
        }
        
    private:
        OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > renderFeeder_;            
        
    }; // class FeedGraphicsPrimitives
    
    
} // anonymous namespace






OpenSteer::Graphics::BatchingRenderFeeder::BatchingRenderFeeder()
: transformedInstancesToRender_(), instancesToRender_(), transformedGraphicsPrimitivesToRender_(), graphicsPrimitivesToRender_(), renderFeederToBatchTo_( new NullRenderFeeder() )
{
    
}



OpenSteer::Graphics::BatchingRenderFeeder::BatchingRenderFeeder( SharedPointer< RenderFeeder > const& _targetFeeder )
: transformedInstancesToRender_(), instancesToRender_(), transformedGraphicsPrimitivesToRender_(), graphicsPrimitivesToRender_(), renderFeederToBatchTo_( 0 != _targetFeeder ? _targetFeeder : SharedPointer< RenderFeeder >( new NullRenderFeeder() ) )
{
    
}



OpenSteer::Graphics::BatchingRenderFeeder::BatchingRenderFeeder( BatchingRenderFeeder const& _other )
    : RenderFeeder( _other ), transformedInstancesToRender_( _other.transformedInstancesToRender_ ), instancesToRender_( _other.instancesToRender_ ), transformedGraphicsPrimitivesToRender_( _other.transformedGraphicsPrimitivesToRender_ ), graphicsPrimitivesToRender_( _other.graphicsPrimitivesToRender_ ), renderFeederToBatchTo_( _other.renderFeederToBatchTo_ )
{
    
}



OpenSteer::Graphics::BatchingRenderFeeder::~BatchingRenderFeeder()
{
    // Nothing to do.
}




OpenSteer::Graphics::BatchingRenderFeeder& 
OpenSteer::Graphics::BatchingRenderFeeder::operator=( BatchingRenderFeeder _other )
{
    swap( _other );
    return *this;
}




void 
OpenSteer::Graphics::BatchingRenderFeeder::swap( BatchingRenderFeeder& _other )
{
    transformedInstancesToRender_.swap( _other.transformedInstancesToRender_ );
    instancesToRender_.swap( _other.instancesToRender_ );
    transformedGraphicsPrimitivesToRender_.swap( _other.transformedGraphicsPrimitivesToRender_ );
    graphicsPrimitivesToRender_.swap( _other.graphicsPrimitivesToRender_ );
    renderFeederToBatchTo_.swap( _other.renderFeederToBatchTo_ );
}





void 
OpenSteer::Graphics::BatchingRenderFeeder::setRenderFeederToBatchTo( SharedPointer< RenderFeeder > const& _renderFeederToBatchTo )
{
    if ( 0 != _renderFeederToBatchTo ) {
        renderFeederToBatchTo_ = _renderFeederToBatchTo;
    } else {
        renderFeederToBatchTo_.reset( new NullRenderFeeder() );
    }
}



OpenSteer::SharedPointer< OpenSteer::Graphics::RenderFeeder > 
OpenSteer::Graphics::BatchingRenderFeeder::renderFeederToBatchTo() const
{
    return renderFeederToBatchTo_;
}




void 
OpenSteer::Graphics::BatchingRenderFeeder::render( Matrix const& _matrix, 
                                                   InstanceId const& _instanceId )
{
    transformedInstancesToRender_.push_back( std::make_pair( _matrix, _instanceId ) );
}



void OpenSteer::Graphics::BatchingRenderFeeder::render( InstanceId const& _instanceId )
{
    instancesToRender_.push_back( _instanceId );
}



void OpenSteer::Graphics::BatchingRenderFeeder::render( Matrix const& _matrix, 
                                                        GraphicsPrimitive const& _graphicsPrimitive )
{
    transformedGraphicsPrimitivesToRender_.push_back( std::make_pair( _matrix, clone( _graphicsPrimitive ) ) );
}



void 
OpenSteer::Graphics::BatchingRenderFeeder::render( GraphicsPrimitive const& _graphicsPrimitive )
{
    graphicsPrimitivesToRender_.push_back( clone( _graphicsPrimitive ) );
}






bool 
OpenSteer::Graphics::BatchingRenderFeeder::addToGraphicsPrimitiveLibrary( GraphicsPrimitive const& _graphicsPrimitive, 
                                                                          InstanceId& _instanceId )
{
    return renderFeederToBatchTo_->addToGraphicsPrimitiveLibrary( _graphicsPrimitive, _instanceId );
}



void 
OpenSteer::Graphics::BatchingRenderFeeder::removeFromGraphicsPrimitiveLibrary( InstanceId const& _instanceId )
{
    renderFeederToBatchTo_->removeFromGraphicsPrimitiveLibrary( _instanceId );
}



bool 
OpenSteer::Graphics::BatchingRenderFeeder::inGraphicsPrimitiveLibrary( InstanceId const& _instanceId ) const
{
    return renderFeederToBatchTo_->inGraphicsPrimitiveLibrary( _instanceId );
}



void 
OpenSteer::Graphics::BatchingRenderFeeder::clearGraphicsPrimitiveLibrary()
{
    renderFeederToBatchTo_->clearGraphicsPrimitiveLibrary();
}





void 
OpenSteer::Graphics::BatchingRenderFeeder::batch()
{
    std::for_each( transformedInstancesToRender_.begin(), 
                   transformedInstancesToRender_.end(), 
                   FeedTransformedInstances( renderFeederToBatchTo_ ) );
    
    std::for_each( instancesToRender_.begin(), 
                   instancesToRender_.end(), 
                   FeedInstances( renderFeederToBatchTo_ ) );
    
    std::for_each( transformedGraphicsPrimitivesToRender_.begin(), 
                   transformedGraphicsPrimitivesToRender_.end(), 
                   FeedTransformedGraphicsPrimitives( renderFeederToBatchTo_ ) );
    
    std::for_each( graphicsPrimitivesToRender_.begin(), 
                   graphicsPrimitivesToRender_.end(), 
                   FeedGraphicsPrimitives( renderFeederToBatchTo_ ) );
    
    
    transformedInstancesToRender_.clear();
    instancesToRender_.clear();
    transformedGraphicsPrimitivesToRender_.clear();
    graphicsPrimitivesToRender_.clear();
}


