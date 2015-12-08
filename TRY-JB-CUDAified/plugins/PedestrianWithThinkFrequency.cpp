// ----------------------------------------------------------------------------
//
//
// OpenSteer -- Steering Behaviors for Autonomous Characters
//
// Copyright (c) 2002-2005, Sony Computer Entertainment America
// Original author: Craig Reynolds <craig_reynolds@playstation.sony.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//
// ----------------------------------------------------------------------------
//
//
// An autonomous "pedestrian":
// follows paths, avoids collisions with obstacles and other pedestrians
//
// 10-29-01 cwr: created
//
//
// ----------------------------------------------------------------------------


/**
 * @todo Don't store a personal random number source in every pedestrian but store one
 * for each pedestrian externally and hand it into the @c simulate member function. On the other
 * hand the random number source might be defined mutable to enable the use of @c const for the
 * @c simulate and @c determineCombinedSteeringVector member functions.
 */

// Include std::exit
#include <cstdlib>

// Include std::priority_queue
#include <queue>

// Include std::size_t
#include <cstddef>

#include <iomanip>
#include <sstream>
#include <iterator>

// Include std::cerr, std::endl
#include <iostream>

// bknafla: @todo Remove if stoping to store steering vectors outside of agents.
// Include std::fill
#include <algorithm>

// Include std::ceil
#include <cmath>

// Include kapaga::random_number_source, kapaga::randomizer< float >
#include "kapaga/randomizer.h"

// Include kapaga::clamp_high
#include "kapaga/math_utilities.h"

// Include KAPAGA_UNUSED_PARAMETER
#include "kapaga/unused_parameter.h"


// Include OpenSteer::ProximityList< AbstractVehicle* , OpenSteer::Vec3>::find_neighbours, std::back_insert_iterator
#include "OpenSteer/PlugInUtilities.h"

#include "OpenSteer/PolylineSegmentedPathwaySingleRadius.h"
#include "OpenSteer/SimpleVehicle.h"
#include "OpenSteer/OpenSteerDemo.h"
// bknafla: Removed the old proximity data structures in exchange for a clearner but perhaps less performant design.
// #include "OpenSteer/Proximity.h"
#include "OpenSteer/Color.h"
#include "OpenSteer/Graphics/GraphicsPrimitives.h"

// Include OpenSteer::Graphics::DrawVehicle, OpenSteer::Graphics::SetAnnotationLogger,
//         OpenSteer::Graphics::SetRenderFeeder
#include "OpenSteer/Graphics/PlugInRenderingUtilities.h"


// Include OpenSteer::GraphicsRenderFeeder
#include "OpenSteer/Graphics/RenderFeeder.h"

// Include OpenSteer::Matrix
#include "OpenSteer/Matrix.h"

// Include OpenSteer::localSpaceTransformationMatrix
#include "OpenSteer/MatrixUtilities.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

#include "OpenSteer/ProximityList.h"

// Include OpenSteer::filter_insert_iterator, OpenSteer::filter_inserter
#include "OpenSteer/filter_insert_iterator.h"



// Include OpenMP functionality.
// @todo Build a wrapper for this.
// #include <omp.h>



namespace {

    using namespace OpenSteer;


    // ----------------------------------------------------------------------------

	// bknafla: Removed. Is this needed anymore?
    // typedef AbstractProximityDatabase<AbstractVehicle*> ProximityDatabase;
	// bknafla: Removed. Is this needed anymore?
    // typedef AbstractTokenForProximityDatabase<AbstractVehicle*> ProximityToken;


    // ----------------------------------------------------------------------------

	// bknafla: Introduce distribution of boid processing over frames.
	// Pedestrians are only "thinking" every @c think_frequency frames. 
	// Not all pedestrians are thinking in 
	// the same frame. The predestrian crowd is partitioned and every 
	// <code> 1 / think_frequency </code>
	// pedestrians are processed in consequtive frames.
	std::size_t const think_frequency = 10;
	

    // How many pedestrians to create when the plugin is opened?
	std::size_t const gPedestrianStartCount = 800; // 400; // 400; // 200; // 100; // 100
    SphereObstacle gObstacle1;
    SphereObstacle gObstacle2;
    ObstacleGroup gObstacles;
    Vec3 gEndpoint0;
    Vec3 gEndpoint1;
    bool gUseDirectedPathFollowing = true;
    // ------------------------------------ xxxcwr11-1-04 fixing steerToAvoid
    RectangleObstacle gObstacle3 (7,7);
    // ------------------------------------ xxxcwr11-1-04 fixing steerToAvoid

    // this was added for debugging tool, but I might as well leave it in
    bool gWanderSwitch = true;

    OpenSteer::SharedPointer< PolylineSegmentedPathwaySingleRadius > gTestPath;

	// Forward declaration.
	// class Pedestrian;
	typedef OpenSteer::ProximityList< AbstractVehicle*, OpenSteer::Vec3 > ProximityContainer;
	
	// How many vehicles to add when pressing the right key?
	std::size_t const add_vehicle_count = 10;
	
} // namespace anonymous


namespace {
	
	
	
    // ---------------------------------------------------------------------------	
	
	
	
	
	
	
    // creates a path for the PlugIn
    // ----------------------------------------------------------------------------
    // create path for PlugIn
    //
    //
    //        | gap |
    //
    //        f      b
    //        |\    /\        -
    //        | \  /  \       ^
    //        |  \/    \      |
    //        |  /\     \     |
    //        | /  \     c   top
    //        |/    \g  /     |
    //        /        /      |
    //       /|       /       V      z     y=0
    //      / |______/        -      ^
    //     /  e      d               |
    //   a/                          |
    //    |<---out-->|               o----> x
    //


    OpenSteer::SharedPointer< PolylineSegmentedPathwaySingleRadius > getTestPath()
    {
        if ( 0 == gTestPath )
        {
            const float pathRadius = 2;

            const PolylineSegmentedPathwaySingleRadius::size_type pathPointCount = 7;
            const float size = 30;
            const float top = 2 * size;
            const float gap = 1.2f * size;
            const float out = 2 * size;
            const float h = 0.5;
            const Vec3 pathPoints[pathPointCount] =
            {Vec3 (h+gap-out,     0,  h+top-out),  // 0 a
                Vec3 (h+gap,         0,  h+top),      // 1 b
                Vec3 (h+gap+(top/2), 0,  h+top/2),    // 2 c
                Vec3 (h+gap,         0,  h),          // 3 d
                Vec3 (h,             0,  h),          // 4 e
                Vec3 (h,             0,  h+top),      // 5 f
                Vec3 (h+gap,         0,  h+top/2)};   // 6 g

            gObstacle1.center = interpolate (0.2f, pathPoints[0], pathPoints[1]);
            gObstacle2.center = interpolate (0.5f, pathPoints[2], pathPoints[3]);
            gObstacle1.radius = 3;
            gObstacle2.radius = 5;
            gObstacles.push_back (&gObstacle1);
            gObstacles.push_back (&gObstacle2);
            // ------------------------------------ xxxcwr11-1-04 fixing steerToAvoid

            gObstacles.push_back (&gObstacle3);

            //         // rotated to be perpendicular with path
            //         gObstacle3.setForward (1, 0, 0);
            //         gObstacle3.setSide (0, 0, 1);
            //         gObstacle3.setPosition (20, 0, h);

            //         // moved up to test off-center
            //         gObstacle3.setForward (1, 0, 0);
            //         gObstacle3.setSide (0, 0, 1);
            //         gObstacle3.setPosition (20, 3, h);

            //         // rotated 90 degrees around path to test other local axis
            //         gObstacle3.setForward (1, 0, 0);
            //         gObstacle3.setSide (0, -1, 0);
            //         gObstacle3.setUp (0, 0, -1);
            //         gObstacle3.setPosition (20, 0, h);

            // tilted 45 degrees
            gObstacle3.setForward (Vec3(1,1,0).normalize());
            gObstacle3.setSide (0,0,1);
            gObstacle3.setUp (Vec3(-1,1,0).normalize());
            gObstacle3.setPosition (20, 0, h);

            //         gObstacle3.setSeenFrom (Obstacle::outside);
            //         gObstacle3.setSeenFrom (Obstacle::inside);
            gObstacle3.setSeenFrom (Obstacle::both);

            // ------------------------------------ xxxcwr11-1-04 fixing steerToAvoid

            gEndpoint0 = pathPoints[0];
            gEndpoint1 = pathPoints[pathPointCount-1];

            gTestPath.reset( new PolylineSegmentedPathwaySingleRadius (pathPointCount,
                                                                       pathPoints,
                                                                       pathRadius,
                                                                       false) );
        }
        return gTestPath;
    }


	int drawPathDirection( kapaga::random_number_source& _rand_source ) {
		return ( kapaga::randomizer< float >::draw( _rand_source ) > 0.5f ) ? -1 : +1;	
	}





    // ---------------------------------------------------------------------------

    class Pedestrian : public SimpleVehicle
    {
    private:
        ProximityContainer const& proximityList_;
		// mutable Vec3 simulatedSteeringForce_;
    public:

        // type for a group of Pedestrians
        typedef std::vector<Pedestrian*> groupType;


        // constructor
        Pedestrian ( ProximityList<AbstractVehicle* , Vec3> const& proximityList, 
					 kapaga::random_number_source const& _rand_source )
			: proximityList_( proximityList ), 
			  rand_source_( _rand_source ),
			  path( getTestPath ().get() ),
			  pathDirection( drawPathDirection( rand_source_ ) )
			/*, simulatedSteeringForce_( Vec3( 0.0f, 0.0f, 0.0f ) ) */ // bknafla: Store the simulated steering forces outside of the agent.
		{

            // pl_=ld;
            // pl_->add(this, position());


/*			// bknafla: remove this
			// Don't store iterators because they can become invalid after adds/removes to the proximitiy data structure.
			pd.add(  ProximityData( this, position() ) );
*/


            // reset Pedestrian state
            reset ( rand_source_ );




        }

        // destructor
        virtual ~Pedestrian ()
        {

            // pl_->remove(this);

/*			// bknafla: remove this
			iter = pd.find_first( this, position() );
			pd.erase( iter );
			// alternative
			pd.remove( this, position );
*/


        }

        // reset all instance state
        void reset ( kapaga::random_number_source const& _rand_source )
        {
            // reset the vehicle
            SimpleVehicle::reset ();

			// Random number state
			rand_source_ = _rand_source;
			
            // max speed and max steering force (maneuverability)
            setMaxSpeed (2.0);
            setMaxForce (8.0);

            // initially stopped
            setSpeed (0);

            // size of bounding sphere, for obstacle avoidance, etc.
            setRadius (0.5); // width = 0.7, add 0.3 margin, take half

            // set the path for this Pedestrian to follow
            path = getTestPath ().get();

            // set initial position
            // (random point on path + random horizontal offset)
            const float d = path->length() * kapaga::randomizer< float >::draw( rand_source_ );
            const float r = path->radius();
            const Vec3 randomOffset = randomVectorOnUnitRadiusXZDisk ( rand_source_ ) * r;
            setPosition (path->mapPathDistanceToPoint (d) + randomOffset);

            // randomize 2D heading
            randomizeHeadingOnXZPlane ( rand_source_ );

            // pick a random direction for path following (upstream or downstream)
            pathDirection = drawPathDirection( rand_source_ );

            // trail parameters: 3 seconds with 60 points along the trail
            setTrailParameters (3, 60);

			// bknafla: Store the simulated steering forces outside of the agent.
			// @todo Uncomment if starting to store steering forces in agents again.
			// simulatedSteeringForce_ = Vec3( 0.0f, 0.0f, 0.0f );
			
            // pl_->update(this, position());

/*			// bknafla: remove me
			pd.update( this, position );
			// alternative
			pd.update( this, old_position, new_position );
			// alternative
			position_iterator iter = pd.find( position );
			if ( *iter == this ) { pd.erase( iter ); }
*/
        }

		Vec3 simulate( float const currentTime, 
					   float const elapsedTime )
		{
			KAPAGA_UNUSED_PARAMETER( currentTime );
			// bknafla: Experiment with not storing the steering vector in the agent
			//          but returning it and storing it outside of it. Mutable just smells bad.
			// simulatedSteeringForce_ = determineCombinedSteering (elapsedTime );
			// return simulatedSteeringForce_;
			// bknafla: Record time statistics here (record_time)
			return determineCombinedSteering (elapsedTime );
		}

        // per frame update
        // updates the position of the pedestrian and draws (some?) annotations

		// bknafla: Experiment with not storing the steering vector in the agent
		//          but returning it and storing it outside of it. Mutable just smells bad.
        void update( float const currentTime, 
					 float const elapsedTime,
					 Vec3 const& simulatedSteeringForce )
        {
			// bknafla: Record time statistics here (record_time)
            // apply steering force to our momentum
            applySteeringForce( simulatedSteeringForce,
								elapsedTime );

            // reverse direction when we reach an endpoint
            if (gUseDirectedPathFollowing)
            {
                const Color darkRed (0.7f, 0, 0);
                float const pathRadius = path->radius();

                if (Vec3::distance (position(), gEndpoint0) < pathRadius )
                {
                    pathDirection = +1;
                    annotationXZCircle (pathRadius, gEndpoint0, darkRed, 20);
                }
                if (Vec3::distance (position(), gEndpoint1) < pathRadius )
                {
                    pathDirection = -1;
                    annotationXZCircle (pathRadius, gEndpoint1, darkRed, 20);
                }
            }

            // annotation
            annotationVelocityAcceleration (5, 0);
            recordTrailVertex (currentTime, position());

			// bknafla: Moved proximity management into the plugin.
            // pl_->update (this, position());

/*			// bknafla: remove me
			pd.update( this, position );
			// alternative
			pd.update( this, old_position, new_position );
			// alternative
			position_iterator iter = pd.find( position );
			if ( *iter == this ) { pd.erase( iter ); }
*/
        }

		/*
        // per frame simulation
        // this does only internal(!) data
        void simulate (const float currentTime, const float elapsedTime)
        {
            // apply steering force to our momentum
            applySteeringForce_simulate (determineCombinedSteering (elapsedTime),
                                elapsedTime);
        }
		 */
		
        // compute combined steering force: move forward, avoid obstacles
        // or neighbors if needed, otherwise follow the path and wander
		// non-const because of random number drawing (wandering behavior)...
        Vec3 determineCombinedSteering ( const float elapsedTime /*,
										 ProximityContainer const& proximityList */ )
        {
            // move forward
            Vec3 steeringForce( forward() );

            // probability that a lower priority behavior will be given a
            // chance to "drive" even if a higher priority behavior might
            // otherwise be triggered.
            float const leakThrough = 0.1f;
			float const random0 = kapaga::randomizer< float >::draw( rand_source_ );
			float const random1 = kapaga::randomizer< float >::draw( rand_source_ );
            // determine if obstacle avoidance is required
            Vec3 obstacleAvoidance;
            if (leakThrough < random0 /* frandom01() */ )
            {
                const float oTime = 6.0f; // minTimeToCollision = 6 seconds
    // ------------------------------------ xxxcwr11-1-04 fixing steerToAvoid
    // just for testing
    //             obstacleAvoidance = steerToAvoidObstacles (oTime, gObstacles);
    //             obstacleAvoidance = steerToAvoidObstacle (oTime, gObstacle1);
    //             obstacleAvoidance = steerToAvoidObstacle (oTime, gObstacle3);
                obstacleAvoidance = steerToAvoidObstacles (oTime, gObstacles);
    // ------------------------------------ xxxcwr11-1-04 fixing steerToAvoid
            }

            // if obstacle avoidance is needed, do it
			steeringForce += obstacleAvoidance;
            if (obstacleAvoidance == Vec3::zero) {
                // otherwise consider avoiding collisions with others
                Vec3 collisionAvoidance;
                

				// @todo Remove this statistics logging!
				/*
				static int call_counter = 0;
				++call_counter;
				static int neighbour_size_accum = 0;
				neighbour_size_accum += neighbours.size();
				std::cout << "call_counter == " << call_counter << " accum == " << neighbour_size_accum << " avg neighbours.size() == " << neighbour_size_accum / call_counter << std::endl;
				*/
				
/*				// bknafla: remove me
				ProximityList pl;
				pd.find_neighbors( position(), maxRadius, pl );
				// std::copy( pl.begin(), pl.end(), neighbors.begin() );
				AVGroup neighbours( pl.begin(), pl.end() );
				// alternative
				pd.find_neighbors( position(), maxRadius, back_inserter_iterator( neighbors ) );
*/

                if (leakThrough < random1 /* frandom01() */ ) {
					const float caLeadTime = 3.0f;
					
					// find all neighbors within maxRadius using proximity database
					// (radius is largest distance between vehicles traveling head-on
					// where a collision is possible within caLeadTime seconds.)
					const float maxRadius = caLeadTime * maxSpeed() * 2.0f;
					
					// @todo Remove magic variable!!
					AVGroup::size_type const neighbourhood_start_init_size = 50;
					AVGroup neighbours;
					neighbours.reserve( neighbourhood_start_init_size );
					 
					// typedef std::priority_queue< OpenSteer::AbstractVehicle*, OpenSteer::AVGroup, OpenSteer::compare_distance > sorted_vector;
					// sorted_vector sorted_neighbours( compare_distance( position() ), neighbours );
					
					
					
					
					
					// std::back_insert_iterator< sorted_vector > backi(sorted_neighbours);
					std::back_insert_iterator< AVGroup > backi( neighbours );
					
					// Find all neighbours that are in a radius of @c maxRadius around the @c Pedestrian @c position().
					// However filter out the pedestrian issuing the search itself using a @c filter_inserter, and the negation (@c std::not1 ) of an equality test ( @c std::equal_to ) with @c this.
					// All pedestrians passing the filter are added to the @c backi output iterator and are therefore added to the @c neighbours @c AVGroup.
					// @attention Because @c backi is a back inserter for an @c AVGroup, which contains pointers to @c AbstractVehicle the type for the @c std::equal_to predicate is also a @c OpenSteer::AbstractVehicle.
					// bknafla: Don't filter itself out here - because the steering behavior to avoid neighbours is already doing this, too...
					proximityList_.find_neighbours( position(), 
													maxRadius,
													backi );
					/*
					 proximityList.find_neighbours( position(), 
													maxRadius,
													filter_inserter( backi, std::not1( std::bind1st( std::equal_to<AVGroup::value_type>(), this ) ) ) );
					 */
					
					
					// bknafla: Added to have a fixed number of neighbours to test against to avoid 
					//          fluctuating performance needs when calculating the steering force.
					// bknafla: @c 7 neighbours as max neighbour count because the brain can
					//          memorize seven items on average.
					// @todo Remove magic variable!
					//std::size_t const max_neighbour_count = 7; // 5;
					/*
					std::sort( neighbours.begin(), neighbours.end(), compare_distance( position() ) );
					if ( neighbours.size() > max_neighbour_count ) {
						neighbours.resize( max_neighbour_count );
					}
					*/
					/*
					std::size_t counter = 0;
					while ( ! sorted_neighbours.empty() &&
							counter < max_neighbour_count ) {
						neighbours.push_back( sorted_neighbours.top() );
						sorted_neighbours.pop();
						++counter;
					}
					*/
					
					
					
                    collisionAvoidance =
                        steerToAvoidNeighbors (caLeadTime, neighbours) * 10.0f;
					
					// if collision avoidance is needed, do it
					// @todo Why not add it all the time? Seems to be cheaper than
					//       a conditional branch, isn't it?
					steeringForce += collisionAvoidance;
				}


                if (collisionAvoidance == Vec3::zero) {
                    // add in wander component (according to user switch)
                    if (gWanderSwitch) {
                        steeringForce += steerForWander (elapsedTime, /* WanderSide, WanderUp, */ rand_source_ );
					}

                    // do (interactively) selected type of path following
                    const float pfLeadTime = 3.0f;
                    const Vec3 pathFollow =
                        (gUseDirectedPathFollowing ?
                         steerToFollowPath (pathDirection, pfLeadTime, *path) :
                         steerToStayOnPath (pfLeadTime, *path));

                    // add in to steeringForce
                    steeringForce += pathFollow * 0.5f;
                }
            }

            // return steering constrained to global XZ "ground" plane
            return steeringForce.setYtoZero ();
        }


        // draw this pedestrian into scene
        void draw()
        {
            // drawBasic2dCircularVehicle (*this, gGray50);

            // annotationLogger().log( Matrix( side(), up(), forward(), position() ), graphicsId_ );

            drawTrail ();
        }


        // called when steerToFollowPath decides steering is required
        virtual void annotatePathFollowing (const Vec3& future,
                                    const Vec3& onPath,
                                    const Vec3& target,
                                    const float outside) const
        {
            const Color yellow (1, 1, 0);
            const Color lightOrange (1.0f, 0.5f, 0.0f);
            const Color darkOrange  (0.6f, 0.3f, 0.0f);
            const Color yellowOrange (1.0f, 0.75f, 0.0f);

            // draw line from our position to our predicted future position
            annotationLine (position(), future, yellow);

            // draw line from our position to our steering target on the path
            annotationLine (position(), target, yellowOrange);

            // draw a two-toned line between the future test point and its
            // projection onto the path, the change from dark to light color
            // indicates the boundary of the tube.
            const Vec3 boundaryOffset = (onPath - future).normalize() * outside;
            const Vec3 onPathBoundary = future + boundaryOffset;
            annotationLine (onPath, onPathBoundary, darkOrange);
            annotationLine (onPathBoundary, future, lightOrange);
        }

        // called when steerToAvoidCloseNeighbors decides steering is required
        // (parameter names commented out to prevent compiler warning from "-W")
        virtual void annotateAvoidCloseNeighbor (const AbstractVehicle& other,
                                         const float /*additionalDistance*/) const
        {
            // draw the word "Ouch!" above colliding vehicles
            float const headOn = forward().dot(other.forward()) < 0;
            Color const green (0.4f, 0.8f, 0.1f);
            Color const red (1, 0.1f, 0);
            Color const color( headOn ? red : green );
            std::string const text( headOn ? "OUCH!" : "pardon me" );
            Vec3 const textOffset( 0.0f, 0.5f, 0.0f );


            // if (OpenSteer::annotationIsOn())
            //    draw2dTextAt3dLocation (*string, location, color, drawGetWindowWidth(), drawGetWindowHeight());

            annotationText( text, localSpaceTransformationMatrix( *this ) * translationMatrix( textOffset ) , color );

        }


        // (parameter names commented out to prevent compiler warning from "-W")
        virtual void annotateAvoidNeighbor (const AbstractVehicle& threat,
                                    const float /*steer*/,
                                    const Vec3& ourFuture,
                                    const Vec3& threatFuture) const
        {
            const Color green (0.15f, 0.6f, 0.0f);

            annotationLine (position(), ourFuture, green);
            annotationLine (threat.position(), threatFuture, green);
            annotationLine (ourFuture, threatFuture, gRed);
            annotationXZCircle (radius(), ourFuture,    green, 12);
            annotationXZCircle (radius(), threatFuture, green, 12);
        }

        // xxx perhaps this should be a call to a general purpose annotation for
        // xxx "local xxx axis aligned box in XZ plane" -- same code in in
        // xxx CaptureTheFlag.cpp
        virtual void annotateAvoidObstacle (const float minDistanceToCollision) const
        {
            const Vec3 boxSide = side() * radius();
            const Vec3 boxFront = forward() * minDistanceToCollision;
            const Vec3 FR = position() + boxFront - boxSide;
            const Vec3 FL = position() + boxFront + boxSide;
            const Vec3 BR = position()            - boxSide;
            const Vec3 BL = position()            + boxSide;
            const Color white (1,1,1);
            annotationLine (FR, FL, white);
            annotationLine (FL, BL, white);
            annotationLine (BL, BR, white);
            annotationLine (BR, FR, white);
        }

        // switch to new proximity database -- just for demo purposes
        void newPD (ProximityContainer* pl)
        {
			KAPAGA_UNUSED_PARAMETER( pl );
			
			// @bknafla: This is done in the plugin for now.
			/*
            pl_->remove(this);
            pl_=pl;
            pl_->add(this, position());
			 */

/*			// bknafla: remove me
			pd_old.remove( this, position );
			pd_new.insert( this, position );
*/



        }

		// Every vehicle has its private random number source. Properly seeded this means that
		// every agent is completely deterministic even when used in a concurrent setting as random
		// numbers aren't a global resource anymore which needs to be locked and which might be
		// called in a different order on different computers because of differences in the 
		// thread scheduling.
		kapaga::random_number_source rand_source_;
		
       // SharedPointer< Graphics::RenderFeeder > renderFeeder_;
       // Graphics::RenderFeeder::InstanceId graphicsId_;

        // path to be followed by this pedestrian
        // XXX Ideally this should be a generic Pathway, but we use the
        // XXX getTotalPathLength and radius methods (currently defined only
        // XXX on PolylinePathway) to set random initial positions.  Could
        // XXX there be a "random position inside path" method on Pathway?
        PolylineSegmentedPathwaySingleRadius* path;

        // direction for path following (upstream or downstream)
        int pathDirection;
    }; // class Pedestrian








// ----------------------------------------------------------------------------



/*
    class DrawVehicle : public std::unary_function< Pedestrian*, void > {
    public:
        DrawVehicle( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder,
                     Graphics::RenderFeeder::InstanceId _vehicleId )
        : renderFeeder_( _renderFeeder ), vehicleId_( _vehicleId ) {

        }

        void operator() ( Pedestrian const* _pedestrian ) {
            renderFeeder_->render( localSpaceTransformationMatrix( *_pedestrian ),
                                   vehicleId_ );
        }

    private:
        SharedPointer< Graphics::RenderFeeder > renderFeeder_;
        Graphics::RenderFeeder::InstanceId vehicleId_;
    }; // class DrawVehicle


    class SetAnnotationLogger : public std::unary_function< Pedestrian*, void > {
    public:
        SetAnnotationLogger( SharedPointer< Graphics::GraphicsAnnotationLogger > _annotationLogger ) : annotationLogger_( _annotationLogger ) {
        }

        void operator()( Pedestrian* _pedestrian ) {
            _pedestrian->setAnnotationLogger( annotationLogger_ );
        }


    private:
        SharedPointer< Graphics::GraphicsAnnotationLogger > annotationLogger_;

    }; // class SetAnnotationLogger


    class SetRenderFeeder : public std::unary_function< Pedestrian*, void > {
    public:
        SetRenderFeeder( SharedPointer< Graphics::RenderFeeder > _renderFeeder ) : renderFeeder_( _renderFeeder ) {
        }

        void operator()( Pedestrian* _pedestrian ) {
            _pedestrian->setRenderFeeder( renderFeeder_ );
        }


    private:
        SharedPointer< Graphics::RenderFeeder > renderFeeder_;

    }; // class SetRenderFeeder
*/







// ----------------------------------------------------------------------------






    // ----------------------------------------------------------------------------
    // OpenSteerDemo PlugIn


    class PedestrianPlugIn : public PlugIn
    {
    private:
        ProximityContainer pl_;
		
		// @todo Move it to @c update. Don't pollute the object with it if it is only needed in one function!
		// bknafla: Experiment with not storing the steering vector in the agent
		//          but returning it and storing it outside of it. Mutable just smells bad.
		// std::vector< Vec3 > steeringForces_;
		
		// bknafla: Introduce thinking frequency.
		std::vector< OpenSteer::Vec3 > simulatedSteeringForces_;
		// bknafla: Introduce thinking frequency.
		std::size_t think_cycle_;
		
		
    public:

        const char* name (void) {return "PedestriansWithThinkingFrequency";}

        float selectionOrderSortKey (void) {return 99.0f;}

        virtual ~PedestrianPlugIn() {}// be more "nice" to avoid a compiler warning

        virtual void open ( SharedPointer< Graphics::RenderFeeder> const& _renderFeeder,
							  SharedPointer< Graphics::GraphicsAnnotationLogger > const& _annotationLogger )
        {
            setAnnotationLogger( _annotationLogger );
            setRenderFeeder( _renderFeeder );

			// bknafla: Introduce thinking frequency.
			think_cycle_ = 0;
			
            getTestPath();


            pedestrianGraphicsId_ = 0;
            obstacle1GraphicsId_  = 0;
            obstacle2GraphicsId_  = 0;
            obstacle3Line0Id_     = 0;
            obstacle3Line1Id_     = 0;
            obstacle3Line2Id_     = 0;
            obstacle3Line3Id_     = 0;
            obstacle3Line4Id_     = 0;

            if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::Vehicle2dGraphicsPrimitive( 0.5f, gGray50 ),
                                                                  pedestrianGraphicsId_ ) ) {
                // @todo Handle this better...
                std::cerr << "Error: Unable to add graphical vehicle representation to render feeder." << std::endl;
            }

            if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::CircleGraphicsPrimitive( gObstacle1.radius, gWhite, 40 ),
                                                                  obstacle1GraphicsId_ ) ) {
                // @todo Handle this better...
                std::cerr << "Error: Unable to add graphical circle representation to render feeder." << std::endl;
            }

            if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::CircleGraphicsPrimitive( gObstacle2.radius, gWhite, 40 ),
                                                                  obstacle2GraphicsId_ ) ) {
                // @todo Handle this better...
                std::cerr << "Error: Unable to add graphical circle representation to render feeder." << std::endl;
            }

            {
                float const w = gObstacle3.width * 0.5f;
                Vec3 const p = gObstacle3.position ();
                Vec3 const s = gObstacle3.side ();

                if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::LineGraphicsPrimitive( p + (s * w), p + (s * -w), gWhite ),
                                                                      obstacle3Line0Id_ )  ) {
                    // @todo Handle this better...
                    std::cerr << "Error: Unable to add graphical line representation to render feeder." << std::endl;
                }

                Vec3 const v1 = gObstacle3.globalizePosition (Vec3 (w, w, 0));
                Vec3 const v2 = gObstacle3.globalizePosition (Vec3 (-w, w, 0));
                Vec3 const v3 = gObstacle3.globalizePosition (Vec3 (-w, -w, 0));
                Vec3 const v4 = gObstacle3.globalizePosition (Vec3 (w, -w, 0));

                if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::LineGraphicsPrimitive( v1, v2, gWhite ),
                                                                      obstacle3Line1Id_ )  ) {
                    // @todo Handle this better...
                    std::cerr << "Error: Unable to add graphical line representation to render feeder." << std::endl;
                }

                if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::LineGraphicsPrimitive( v2, v3, gWhite ),
                                                                      obstacle3Line2Id_ )  ) {
                    // @todo Handle this better...
                    std::cerr << "Error: Unable to add graphical line representation to render feeder." << std::endl;
                }

                if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::LineGraphicsPrimitive( v3, v4, gWhite ),
                                                                      obstacle3Line3Id_ )  ) {
                    // @todo Handle this better...
                    std::cerr << "Error: Unable to add graphical line representation to render feeder." << std::endl;
                }

                if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( Graphics::LineGraphicsPrimitive( v4, v1, gWhite ),
                                                                      obstacle3Line4Id_ )  ) {
                    // @todo Handle this better...
                    std::cerr << "Error: Unable to add graphical line representation to render feeder." << std::endl;
                }

            }



            // make the database used to accelerate proximity queries
            cyclePD = -1;
            nextPD ();

            // create the specified number of Pedestrians
			// @todo: Move the functionality to add or remove more than one pedestrain into the add and remove member functions.
            population = 0;
            for ( std::size_t i = 0; i < gPedestrianStartCount; ++i ) {
				addPedestrianToCrowd ();
			}

            // initialize camera and selectedVehicle
            Pedestrian& firstPedestrian = **crowd.begin();
            OpenSteerDemo::init3dCamera (firstPedestrian);
            OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
            OpenSteerDemo::camera.fixedTarget.set (15, 0, 30);
            OpenSteerDemo::camera.fixedPosition.set (15, 70, -70);
        }


		
        virtual void update( const float currentTime, const float elapsedTime )
        {
			// First let all pedestrian simulate their next step
			// Second: update the pedestrian with the simulation results.
			// Use these data to feed the physics system and get the physically 
			// correct pedestrian positions and orientation. (This step is currently missing)
			// Then update the proximity database with their new positions
			
			// bknafla: Record time statistics here (record_time)
			// kapaga::omp_timer full_update_timer;
			
			
			// std::vector< Vec3 > steeringForces( crowd.size(), Vec3::zero );
			
			

			
		
			
			int const crowd_size = static_cast< int >( crowd.size() );
			
			// bknafla: Determine which boids to simulate in this frame.
			think_cycle_ = ( think_cycle_ + 1 ) % think_frequency;
			int const thinking_vehicle_count_per_frame = static_cast< int >( std::ceil( static_cast< float >( crowd_size ) / static_cast< float >( think_frequency ) ) );
			int const think_lower_vehicle_index = think_cycle_ * thinking_vehicle_count_per_frame;
			int const think_higher_vehicle_index = kapaga::clamp_high( think_lower_vehicle_index + thinking_vehicle_count_per_frame, crowd_size );
			
			
			#pragma omp parallel default(shared) 
			{
				
				// @todo Remove after first tests (otherwise there should be a critical section
				//       or a lock around the usage of the global resource @c std::clog).
				// std::clog << "In parallel region: " << omp_in_parallel() << std::endl;
				// std::clog << "omp_get_num_threads: " << omp_get_num_threads() << std::endl;
				
			// Simulate the steering without changing the observable agent state.
			#pragma omp for schedule(dynamic)
			for ( int i = think_lower_vehicle_index; i < think_higher_vehicle_index; ++i ) {
				// Calculate the steering force of every agent (that depends on the neighborhood to 
				// other agents and store the simulated steering forces outside of the agent.
				// This way no agent is changed while the simulation is still running and no agent 
				// can see the future position of another agent.
				
				// kapaga::omp_timer timer;
				simulatedSteeringForces_[ i ] = crowd[ i ]->simulate( currentTime, elapsedTime );
				// simulation_timer_thread_storage.slot().push_back( info( i, timer.elapsed_time() ) );
			}
			
			// Apply the simulation results to update the agent states
			//  schedule( dynamic, 100 )
			#pragma omp  for schedule(static)
			for ( int i = 0; i < crowd_size; ++i ) {
				// Update each agent with its calculated steering force. The agents state is also 
				// part of the world state.
				// kapaga::omp_timer timer;
				// typedef kapaga::omp_timer:.time_type time_type;
				crowd[ i ]->update( currentTime, elapsedTime, simulatedSteeringForces_[ i ] );
				// time_type after_vehicles_before_proximit_update_time = timer.elapsed_time();
				
				// Update the proximity database with the new agent positions. 
				
				// kapaga::omp_timer proximity_update_timer;
				pl_.update( crowd[ i ], ( crowd[ i ] )->position() );
				// timer_type full_update_of_vehicle_and_proximity_timer = timer.elapsed_time();
				// log these times + full time
			}
			

			// Let the physics system do its magic to calculate the physically correct next
			// position, orientation, speed, acceleration, etc. of the agents.
			// ... missing...
			
			
			// Update the proximity database with the new agent positions. 
			/* Commented because these calcs are now done together with the agent update above.
			#pragma omp  for default(shared)
			for ( int i = 0; i < crowd_size; ++i ) {
				pl_.update( crowd[ i ], ( crowd[ i ] )->position() );
			}
			*/
			}
			 
			
			
        }

        /**
         * Draws all pedestrians. Currently only the graphical vehicle
         * representations are drawn and @c draw has to be called for every
         * vehicle to draw its trail.
         */
        void drawCrowd() {

            std::for_each( crowd.begin(), crowd.end(), OpenSteer::Graphics::DrawVehicle<Pedestrian>( renderFeeder(), pedestrianGraphicsId_ ) );
			/*
			int const crowd_count = static_cast< int >( crowd.size() );
			#pragma omp parallel for default( shared )
			for ( int i = 0; i < crowd_count; ++i ) {
				renderFeeder()->render( localSpaceTransformationMatrix( *( crowd[ i ] ) ),
										pedestrianGraphicsId_ );
			}
			 */

        }


        virtual void redraw (const float currentTime, const float elapsedTime)
        {
			// bknafla: Record time statistics here (record_time)
			
			KAPAGA_UNUSED_PARAMETER( currentTime );
			KAPAGA_UNUSED_PARAMETER( elapsedTime );
			
            // selected Pedestrian (user can mouse click to select another)
            AbstractVehicle& selected = *OpenSteerDemo::selectedVehicle;

            // Pedestrian nearest mouse (to be highlighted)
            AbstractVehicle& nearMouse = *OpenSteerDemo::vehicleNearestToMouse ();

            // update camera
            // OpenSteerDemo::updateCamera (currentTime, elapsedTime, selected);

            // draw "ground plane"
            if (OpenSteerDemo::selectedVehicle) gridCenter = selected.position();
            OpenSteerDemo::gridUtility (gridCenter);


			// The code of drawCrowd is inserted into the parallel loop below to deploy having
			// the vehicle data in the cache while also using parallelization.
			drawCrowd();

			// bknafla: Make use of the parallel render feeders!!
			// draw and annotate each Pedestrian
			//int const crowd_count = static_cast< int >( crowd.size() );
			//#pragma omp parallel for default( shared )
			//for ( int i = 0; i < crowd_count; ++i ) {
			for ( iterator i = crowd.begin(); i != crowd.end(); ++i) {
				// Draw footsteps.
				(*i)->draw ();
				//crowd[ i ]->draw();
				// Draw pedestrian shape.
				//renderFeeder()->render( localSpaceTransformationMatrix( *( crowd[ i ] ) ),
				//						pedestrianGraphicsId_ );
			}
		
			// draw the path they follow and obstacles they avoid
			drawPathAndObstacles ();

					
            // highlight Pedestrian nearest mouse
            OpenSteerDemo::highlightVehicleUtility (nearMouse);

            // textual annotation (at the vehicle's screen position)
            serialNumberAnnotationUtility (selected, nearMouse);

            // textual annotation for selected Pedestrian
            if (OpenSteerDemo::selectedVehicle && OpenSteer::annotationIsOn())
            {
                const Color color (0.8f, 0.8f, 1.0f);
                const Vec3 textOffset (0, 0.25f, 0);
                // const Vec3 textPosition = selected.position() + textOffset;
                const Vec3 camPosition = OpenSteerDemo::camera.position();
                const float camDistance = Vec3::distance (selected.position(),
                                                          camPosition);
                const char* spacer = "      ";
                std::ostringstream annote;
                annote << std::setprecision (2);
                annote << std::setiosflags (std::ios::fixed);
                annote << spacer << "1: speed: " << selected.speed() << std::endl;
                annote << std::setprecision (1);
                annote << spacer << "2: cam dist: " << camDistance << std::endl;
                annote << spacer << "3: no third thing" << std::ends;
                // draw2dTextAt3dLocation (annote, textPosition, color, drawGetWindowWidth(), drawGetWindowHeight());
                renderFeeder()->render( localSpaceTransformationMatrix( selected ) * translationMatrix( textOffset ), Graphics::TextAt3dLocationGraphicsPrimitive( annote.str(), color ) );

            }

            // display status in the upper left corner of the window
            std::ostringstream status;
            status << "[F1/F2] Crowd size: " << population;
            status << "\n[F3] PD type: ";
            switch (cyclePD)
            {
            case 0: status << "LQ bin lattice"; break;
            case 1: status << "brute force";    break;
            }
            status << "\n[F4] ";
            if (gUseDirectedPathFollowing)
                status << "Directed path following.";
            else
                status << "Stay on the path.";
            status << "\n[F5] Wander: ";
            if (gWanderSwitch) status << "yes"; else status << "no";
            status << std::endl;
            // float const h = drawGetWindowHeight ();
            Vec3 const screenLocation ( 10.0f, 50.0f, 0.0f );
            // draw2dTextAt2dLocation (status, screenLocation, gGray80, drawGetWindowWidth(), drawGetWindowHeight());


            renderFeeder()->render( Graphics::TextAt2dLocationGraphicsPrimitive( screenLocation, Graphics::TextAt2dLocationGraphicsPrimitive::TOP_LEFT, status.str(), gGray80 ) );
        }


        void serialNumberAnnotationUtility (const AbstractVehicle& selected,
                                            const AbstractVehicle& nearMouse)
        {
            // display a Pedestrian's serial number as a text label near its
            // screen position when it is near the selected vehicle or mouse.
            if (&selected && &nearMouse && OpenSteer::annotationIsOn())
            {
				// int const crowd_count = static_cast< int > (crowd.size() );
				// #pragma omp parallel for default( shared )
				// for ( int i = 0; i < crowd_count; ++i )
                for (iterator i = crowd.begin(); i != crowd.end(); ++i)
                {
                    AbstractVehicle const* vehicle = *i;
					//AbstractVehicle const* vehicle = ( crowd[ i ] );
                    const float nearDistance = 6;
                    const Vec3& vp = vehicle->position();
                    const Vec3& np = nearMouse.position();
                    if ((Vec3::distance (vp, selected.position()) < nearDistance)
                        ||
                        (&nearMouse && (Vec3::distance (vp, np) < nearDistance)))
                    {
                        std::ostringstream sn;
                        sn << "#"
                           << ((Pedestrian*)vehicle)->serialNumber
                           << std::ends;
                        const Color textColor (0.8f, 1.0f, 0.8f);
                        const Vec3 textOffset (0, 0.25f, 0);
                        // const Vec3 textPos = vehicle->position() + textOffset;
                        // draw2dTextAt3dLocation (sn, textPos, textColor, drawGetWindowWidth(), drawGetWindowHeight());
                        renderFeeder()->render( localSpaceTransformationMatrix( *vehicle ) * translationMatrix( textOffset ), Graphics::TextAt3dLocationGraphicsPrimitive( sn.str(), textColor ) );
                    }
                }
            }
        }

        void drawPathAndObstacles (void)
        {
            typedef PolylineSegmentedPathwaySingleRadius::size_type size_type;

            // draw a line along each segment of path
            const PolylineSegmentedPathwaySingleRadius& path = *getTestPath ();
			
			// @todo Remove if thread storage isn't used anymore!
			int const point_count = static_cast< int >( path.pointCount() );
			// #pragma omp parallel for default( shared )
			for (int i = 1; i < point_count; ++i ) {
            // for (size_type i = 1; i < path.pointCount(); ++i ) {
                // drawLine (path.point( i ), path.point( i-1) , gRed);
                renderFeeder()->render( Graphics::LineGraphicsPrimitive( path.point( static_cast< size_type >( i ) ), 
																		 path.point( static_cast< size_type >( i - 1 )  ), gRed ) );
            }

            renderFeeder()->render( translationMatrix( gObstacle1.center ), obstacle1GraphicsId_ );
            renderFeeder()->render( translationMatrix( gObstacle2.center ), obstacle2GraphicsId_ );
            renderFeeder()->render( obstacle3Line0Id_ );
            renderFeeder()->render( obstacle3Line1Id_ );
            renderFeeder()->render( obstacle3Line2Id_ );
            renderFeeder()->render( obstacle3Line3Id_ );
            renderFeeder()->render( obstacle3Line4Id_ );
        }

        virtual void close (void)
        {
            // delete all Pedestrians
            while ( population > 0 ) {
				removePedestrianFromCrowd ();
			}

            renderFeeder()->removeFromGraphicsPrimitiveLibrary( pedestrianGraphicsId_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle1GraphicsId_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle2GraphicsId_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle3Line0Id_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle3Line1Id_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle3Line2Id_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle3Line3Id_ );
            renderFeeder()->removeFromGraphicsPrimitiveLibrary( obstacle3Line4Id_ );
        }

        virtual void reset (void)
        {
            // reset each Pedestrian
			/* parallelize...
            for (iterator i = crowd.begin(); i != crowd.end(); ++i ) {
				(*i)->reset ();
				pl.update( *i, (*i)->position() );
			}
			 */
			
			int const crowd_size = static_cast< int >( crowd.size() );
			// @attention Drawing random numbers to place the pedestrians isn't thread safe!!!!
			// #pragma omp parallel for default(shared) 
			for ( int i = 0; i < crowd_size; ++i ) {
				// Vehicles determine their position using a random number source. 
				// They copy the random number source of the plugin which is changed 
				// (a random number is drawn for every boid) so each boid gets a 
				// different first random number from it.
				KAPAGA_UNUSED_RETURN_VALUE( rand_source_.draw() );
				crowd[ i ]->reset( rand_source_ );
				pl_.update( crowd[ i ], crowd[ i ]->position() );
			}
			
			// bknafla: @c steeringForces_ is no local to the only member function using it.
			// bknafla: Store the simulated steering forces outside of the agent.
			// std::fill( steeringForces_.begin(), steeringForces_.end(), Vec3( 0.0f, 0.0f, 0.0f ) );

			// bknafla: Introduce thinking frequency.
			think_cycle_ = 0;
			for ( std::vector< Vec3 >::iterator i = simulatedSteeringForces_.begin(); i != simulatedSteeringForces_.end(); ++i ) {
				*i = OpenSteer::randomVectorOnUnitRadiusXZDisk( rand_source_ );
			}
			
			
			
            // reset camera position
            OpenSteerDemo::position2dCamera (*OpenSteerDemo::selectedVehicle);

            // make camera jump immediately to new position
            OpenSteerDemo::camera.doNotSmoothNextMove ();
        }
		

        void handleFunctionKeys (int keyNumber)
        {
            switch (keyNumber)
            {
				case 1:  for ( std::size_t i = 0; i < add_vehicle_count; ++i) { addPedestrianToCrowd ();    }                           break;
				case 2:  for ( std::size_t i = 0; i < add_vehicle_count; ++i ) { removePedestrianFromCrowd ();    }                      break;
				case 3:  nextPD ();                                             break;
				case 4: gUseDirectedPathFollowing = !gUseDirectedPathFollowing; break;
				case 5: gWanderSwitch = !gWanderSwitch;                         break;
				default:
					break;
            }
        }

        void printMiniHelpForFunctionKeys (void)
        {
            std::ostringstream message;
            message << "Function keys handled by ";
            message << '"' << name() << '"' << ':' << std::ends;
            OpenSteerDemo::printMessage (message);
            OpenSteerDemo::printMessage (message);
            OpenSteerDemo::printMessage ("  F1     add a pedestrian to the crowd.");
            OpenSteerDemo::printMessage ("  F2     remove a pedestrian from crowd.");
            OpenSteerDemo::printMessage ("  F3     use next proximity database.");
            OpenSteerDemo::printMessage ("  F4     toggle directed path follow.");
            OpenSteerDemo::printMessage ("  F5     toggle wander component on/off.");
            OpenSteerDemo::printMessage ("");
        }


        void addPedestrianToCrowd (void)
        {
			// @todo: Check for exception-safeness.
			rand_source_.draw();
            Pedestrian* pedestrian = new Pedestrian( pl_, rand_source_ ); // &pl /*, pedestrianGraphicsId_ */ );
            pedestrian->setRenderFeeder( renderFeeder() );
            pedestrian->setAnnotationLogger( annotationLogger() );
            crowd.push_back (pedestrian);
			// bknafla: introduce thinking frequency.
			simulatedSteeringForces_.push_back( OpenSteer::randomVectorOnUnitRadiusXZDisk( rand_source_ ) );
			++population;
			
			// bknafla. @c steeringForces_ is now local to the only member function using it.
			// bknafla: Store the simulated steering forces outside of the agent.
			// steeringForces_.push_back( Vec3( 0.0f, 0.0f, 0.0f ) );
			
			pl_.add( pedestrian, pedestrian->position() );
			
            if (population == 1) {
				OpenSteerDemo::selectedVehicle = pedestrian;
			}
        }


        void removePedestrianFromCrowd (void)
        {
            if (population > 0)
            {
                // save pointer to last pedestrian, then remove it from the crowd
				Pedestrian* pedestrian = crowd.back();
                crowd.pop_back();
				// bknafla: introduce thinking frequency.
				simulatedSteeringForces_.pop_back();
                --population;

                // if it is OpenSteerDemo's selected vehicle, unselect it
                if (pedestrian == OpenSteerDemo::selectedVehicle) {
                    OpenSteerDemo::selectedVehicle = NULL;
				}

				// bknafla. @c steeringForces_ is now local to the only member function using it.
				// bknafla: Store the simulated steering forces outside of the agent.
				// steeringForces_.pop_back();
				
                // delete the Pedestrian
				pl_.remove( pedestrian );
                delete pedestrian;
            }
        }


        // for purposes of demonstration, allow cycling through various
        // types of proximity databases.  this routine is called when the
        // OpenSteerDemo user pushes a function key.
        void nextPD (void)
        {
/*            // save pointer to old PD
            ProximityDatabase* oldPD = pd;

            // allocate new PD
            const int totalPD = 2;
            switch (cyclePD = (cyclePD + 1) % totalPD)
            {
            case 0:
                {
                    const Vec3 center;
                    const float div = 20.0f;
                    const Vec3 divisions (div, 1.0f, div);
                    const float diameter = 80.0f; //XXX need better way to get this
                    const Vec3 dimensions (diameter, diameter, diameter);
                    typedef LQProximityDatabase<AbstractVehicle*> LQPDAV;
                    pd = new LQPDAV (center, dimensions, divisions);
                    break;
                }
            case 1:
                {
                    pd = new BruteForceProximityDatabase<AbstractVehicle*> ();
                    break;
                }
            }

            // switch each boid to new PD
            for (iterator i=crowd.begin(); i!=crowd.end(); i++) (**i).newPD(*pd);

            // delete old PD (if any)
            delete oldPD;*/
        }


        virtual void setRenderFeeder( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder ) {
            PlugIn::setRenderFeeder( _renderFeeder );

            std::for_each( crowd.begin(), crowd.end(), OpenSteer::Graphics::SetRenderFeeder< Pedestrian >( _renderFeeder ) );
        }

        virtual void setAnnotationLogger( SharedPointer< Graphics::GraphicsAnnotationLogger > const& _annotationLogger ) {
            PlugIn::setAnnotationLogger( _annotationLogger );

            std::for_each( crowd.begin(), crowd.end(), OpenSteer::Graphics::SetAnnotationLogger< Pedestrian >( _annotationLogger ) );
        }



        Graphics::RenderFeeder::InstanceId pedestrianGraphicsId_;
        Graphics::RenderFeeder::InstanceId obstacle1GraphicsId_;
        Graphics::RenderFeeder::InstanceId obstacle2GraphicsId_;
        Graphics::RenderFeeder::InstanceId obstacle3Line0Id_;
        Graphics::RenderFeeder::InstanceId obstacle3Line1Id_;
        Graphics::RenderFeeder::InstanceId obstacle3Line2Id_;
        Graphics::RenderFeeder::InstanceId obstacle3Line3Id_;
        Graphics::RenderFeeder::InstanceId obstacle3Line4Id_;

        // Graphics::AnnotationScene* annotationScene_;

        const AVGroup& allVehicles (void) {return (const AVGroup&) crowd;}

        // crowd: a group (STL vector) of all Pedestrians
        Pedestrian::groupType crowd;
        typedef Pedestrian::groupType::const_iterator iterator;

        Vec3 gridCenter;

		// bknafla: removed: use @c pl (proximity list) instead.
        // pointer to database used to accelerate proximity queries
        // ProximityDatabase* pd;

        // keep track of current flock size
        int population;

        // which of the various proximity databases is currently in use
        int cyclePD;
		
		kapaga::random_number_source rand_source_;
    };


    PedestrianPlugIn gPedestrianPlugIn;




    // ----------------------------------------------------------------------------

} // anonymous namespace

