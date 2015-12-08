// @todo Remove after debugging
// Include std::clog, std::endl
#include <iostream>
// Include std::numeric_limits<float>::quiet_NaN
#include <limits>


// Include std::size_t
#include <cstddef>

#include <sstream>

// bknafla: adapt to thread safe random number generation.
// Include kapaga::random_number_source
#include "kapaga/random_number_source.h"

#include "OpenSteer/SimpleVehicle.h"
#include "OpenSteer/OpenSteerDemo.h"
// bknafla: Introduce parallelization. Replace proximity data structures.
// #include "OpenSteer/Proximity.h"
#include "OpenSteer/ProximityList.h"
#include "OpenSteer/Color.h"

// bknafla: Change render interface to adapt to parallelization.
// Include OpenSteer::Graphics::DrawVehicle, OpenSteer::Graphics::SetAnnotationLogger,
//         OpenSteer::Graphics::SetRenderFeeder
#include "OpenSteer/Graphics/PlugInRenderingUtilities.h"

// Include OpenSteer::ProximityList< AbstractVehicle* , OpenSteer::Vec3>::find_neighbours, std::back_insert_iterator
#include "OpenSteer/PlugInUtilities.h"

// bknafla: Change render interface to adapt to parallelization.
//Include OpenSteer::Graphics::makeBasic3dSphericalVehicleGraphicsPrimitive
#include "OpenSteer/Graphics/GraphicsPrimitivesUtilities.h"

// Include OpenSteer::localSpaceTransformationMatrix, OpenSteer::translationMatrix
#include "OpenSteer/MatrixUtilities.h"



namespace {
	// Include names declared in the OpenSteer namespace into the
	// namespaces to search to find names.
	using namespace OpenSteer;

	// How many boids to create when the plugin is opened?
	std::size_t const gBoidStartCount = 4032; // 400; // 400; // 200; // 100;


	class Boid;  // Forward declaration.
	typedef OpenSteer::ProximityList< Boid*, OpenSteer::Vec3 > ProximityContainer;
	
	// How many vehicles to add when pressing the right key?
	std::size_t const add_vehicle_count = 10;



/***  MAGIC VARIABLES FOR BOIDS CLASS  ***/
const float worldRadius      = 50.0f;
const float separationRadius =  5.0f;
const float separationAngle  = -0.707f;
const float separationWeight =  12.0f;

const float alignmentRadius  = 7.5f;
const float alignmentAngle   = 0.7f;
const float alignmentWeight  = 8.0f;

const float cohesionRadius   = 9.0f;
const float cohesionAngle    = -0.15f;
const float cohesionWeight   = 8.0f;

const std::size_t neighbour_size_max = 7;


class Boid {
	public:
		// group of all obstacles to be avoided by each Boid
		static ObstacleGroup obstacles;

		// type for a flock: an STL vector of Boid pointers
		typedef std::vector<Boid>        groupType;
		typedef std::vector<Boid*>       AVGroup;
		typedef AVGroup::const_iterator  AVIterator;
		

		/**
		 * Default constructor
		 * @param pcontainer The proximity datastructure this boid lives in
		 */
		Boid(kapaga::random_number_source const& _rand_source) {
			reset(_rand_source);
		}

		/**
		 * Copy Constructor
		 */
		Boid(const Boid &c) :
		position_(c.position_), up_(c.up_), side_(c.side_), forward_(c.forward_), smoothedAcceleration_(c.smoothedAcceleration_), smoothedPosition_(c.smoothedPosition_), lastForward_(c.lastForward_), lastPosition_(c.lastPosition_), speed_(c.speed_), curvature_(c.curvature_), smoothedCurvature_(c.smoothedCurvature_)
		{ }

		void reset (kapaga::random_number_source const &_rand_source) {
			forward_.set (0, 0, 1);
			speed_ = maxSpeed_ * 0.3f;

			smoothedAcceleration_ = Vec3::zero;

			side_ = Vec3 (- forward_.z,  forward_.y, forward_.x);
			up_.set (0, 1, 0);

			// Only @c reset needs to draw random numbers in some function calls.
			kapaga::random_number_source rand_source( _rand_source );

			// bknafla: adapt to thread safe random number generation.
			// randomize initial orientation
			regenerateOrthonormalBasisUF( RandomUnitVector ( rand_source ) );

			// randomize initial position
			setPosition (RandomVectorInUnitRadiusSphere( rand_source ) * 20.0f);
		}


		/**
		 * Simulate where we want to be next
		 */
		Vec3 simulate(const ProximityContainer &pc ) {
			// avoid obstacles if needed
			/// @todo
			//const Vec3 avoidance = steerToAvoidObstacles (1.0f, obstacles);

			//if (avoidance != Vec3::zero) {
			//	return avoidance;
			//}


			static const float maxRadius = maxXXX (separationRadius,
						               maxXXX (alignmentRadius, cohesionRadius)
						              );


			AVGroup neighbours;
			neighbours.reserve( neighbour_size_max );

			std::back_insert_iterator< std::vector<Boid*> > backi(neighbours);
			pc.find_neighbours( position(), maxRadius, backi );

			// determine each of the three component behaviors of flocking
			const Vec3 separation  = steerForSeparation (separationRadius*separationRadius, separationAngle, neighbours);
			const Vec3 alignment   = steerForAlignment  (alignmentRadius*alignmentRadius, alignmentAngle, neighbours);
			const Vec3 cohesion    = steerForCohesion   (cohesionRadius*cohesionRadius, cohesionAngle, neighbours);

			// apply weights to components (save in variables for annotation)
			const Vec3 separationW = separation * separationWeight;
			const Vec3 alignmentW  = alignment * alignmentWeight;
			const Vec3 cohesionW   = cohesion * cohesionWeight;

			return separationW + alignmentW + cohesionW;
		}

		/**
		 * per frame simulation update
		 */
		void update ( float const elapsedTime, Vec3 const& _simulatedSteeringForce ) {
			applySteeringForce ( _simulatedSteeringForce, elapsedTime);
			sphericalWrapAround ();
		}




		/**
		 * get our internal state :-)
		 */
		Vec3  position() const  { return position_;         }
		Vec3  side    () const  { return side_;             }
		Vec3  up      () const  { return up_;               }
		Vec3  forward () const  { return forward_;          }


	private:
		Vec3  velocity() const  { return forward_ * speed_; }
		float radius  () const  { return radius_;           }
		float maxForce() const  { return maxForce_;         }
		float maxSpeed() const  { return maxSpeed_;         }
		float mass    () const  { return mass_;             }
		float speed   () const  { return speed_;            }
		
		/**
		 * set our internal state :-(
		 */
		void setPosition(const Vec3 &v) { position_ = v; }
		void setUp      (const Vec3 &v) { up_       = v; }
		void setForward (const Vec3 &v) { forward_  = v; }
		void setSpeed   (const float v) { speed_    = v; }
		
		/**
		 * constrain this boid to stay within sphereical boundary.
		 */
		void sphericalWrapAround (void) {
			// when outside the sphere
			if (position().length() > worldRadius) {
				// wrap around (teleport)
				setPosition (position().sphericalWrapAround (Vec3::zero, worldRadius));
			}
		}

		/**
		 * Seperation :-)
		 */
		Vec3 steerForSeparation (const float maxDistance, const float cosMaxAngle, const AVGroup& flock) const {
			static const float minDistanceSquared = radius()*3;
			// steering accumulator and count of neighbors, both initially zero
			Vec3 steering( Vec3::zero );

			// for each of the other vehicles...
			AVIterator flockEndIter = flock.end();
			for (std::size_t i=0; i<neighbour_size_max; ++i) {
				if (flock[i]==0) break;
				const Boid &cur_neighbor = *flock[i];
				if (inBoidNeighborhood (cur_neighbor, minDistanceSquared, maxDistance, cosMaxAngle)) {

					const Vec3 offset = cur_neighbor.position() - position();

					const float distanceSquared = offset.lengthSquared();

					if ( 0.0f != distanceSquared ) {
						steering -= (offset / distanceSquared);
					} else {
						steering -= offset;
					}

				}
			}

			return steering.normalize();
		}

		/**
		 * Alignment
		 */
		Vec3 steerForAlignment (const float maxDistance, const float cosMaxAngle, const AVGroup& flock) const {
			static const float minDistanceSquared = radius()*3;
			// steering accumulator and count of neighbors, both initially zero
			Vec3 steering( Vec3::zero );
			int influencing_neighbour_count = 0;

			// for each of the other vehicles...
			for (std::size_t i=0; i<neighbour_size_max; ++i) {
				if (flock[i]==0) break;
				const Boid &cur_neighbor = *flock[i];
				if (inBoidNeighborhood (cur_neighbor, minDistanceSquared, maxDistance, cosMaxAngle)) {
					// accumulate sum of neighbor's heading
					steering += cur_neighbor.forward();

					// count neighbors
					++influencing_neighbour_count;
				}
			}

			steering -= ( forward() * influencing_neighbour_count );

			return steering.normalize();
		}

		/**
		 * Cohesion
		 */
		Vec3 steerForCohesion (const float maxDistance, const float cosMaxAngle, const AVGroup& flock) const {
			static const float minDistanceSquared = radius()*3;
			// steering accumulator and count of neighbors, both initially zero
			Vec3 steering( Vec3::zero );
			int influencing_neighbour_count = 0;

			// for each of the other vehicles...
			for (std::size_t i=0; i<neighbour_size_max; ++i) {
				if (flock[i]==0) break;
				const Boid &cur_neighbor = *flock[i];
				if (inBoidNeighborhood (cur_neighbor, minDistanceSquared, maxDistance, cosMaxAngle)) {
					// accumulate sum of neighbor's positions
					steering += cur_neighbor.position();

					// count neighbors
					++influencing_neighbour_count;
				}
			}

			steering -= ( position() * influencing_neighbour_count );


			return steering.normalize();
		}

		/**
		 * A neighbor detection
		 */
		bool inBoidNeighborhood (const Boid& otherVehicle, const float minDistanceSquared, const float maxDistanceSquared, const float cosMaxAngle) const {
			if (&otherVehicle == this) {
				return false;
			}
			
			const Vec3 offset           = otherVehicle.position() - position();
			const float distanceSquared = offset.lengthSquared ();

			// definitely in neighborhood if inside minDistance sphere
			if (distanceSquared < minDistanceSquared) {
				return true;
			}

			if (distanceSquared > maxDistanceSquared) {
				return false;
			}
			
			// otherwise, test angular offset from forward axis
			const Vec3  unitOffset  = offset / sqrt (distanceSquared);
			const float forwardness = forward().dot (unitOffset);
			return forwardness > cosMaxAngle;
		}

		/**
		 * Apply the data from the last update
		 */
		void applySteeringForce( Vec3 const& force, float const elapsedTime) {
			const Vec3 adjustedForce = adjustRawSteeringForce (force);

			// enforce limit on magnitude of steering force
			const Vec3 clippedForce = adjustedForce.truncateLength (maxForce ());

			// compute acceleration and velocity
			Vec3 const newAcceleration = (clippedForce / mass());
			Vec3 newVelocity = velocity();

			// damp out abrupt changes and oscillations in steering acceleration
			// (rate is proportional to time step, then clipped into useful range)
			if (elapsedTime > 0) {
				const float smoothRate = clip (9.0f * elapsedTime, 0.15f, 0.4f);
				blendIntoAccumulator (smoothRate, newAcceleration, smoothedAcceleration_);
			}

			// Euler integrate (per frame) acceleration into velocity
			newVelocity += smoothedAcceleration_ * elapsedTime;

			// enforce speed limit
			newVelocity = newVelocity.truncateLength (maxSpeed ());

			// update Speed
			setSpeed ( newVelocity.length() );

			// Euler integrate (per frame) velocity into position
			setPosition (position() + (newVelocity * elapsedTime));

			// regenerate local space (by default: align vehicle's forward axis with
			// new velocity, but this behavior may be overridden by derived classes.)
			regenerateLocalSpace (newVelocity, elapsedTime);

			// maintain path curvature information
			measurePathCurvature (elapsedTime);

			// running average of recent positions
			blendIntoAccumulator (elapsedTime * 0.06f, position (), smoothedPosition_);
		}

		/**
		 * I hear you calling
		 */
		void regenerateLocalSpace (const Vec3& newVelocity, const float elapsedTime ) {
			// 3d flight with banking
			regenerateLocalSpaceForBanking (newVelocity, elapsedTime);
		}

		/**
		 * Yeeeah, you know me
		 */
		void regenerateLocalSpaceForBanking (const Vec3& newVelocity, const float elapsedTime) {
			// the length of this global-upward-pointing vector controls the vehicle's
			// tendency to right itself as it is rolled over from turning acceleration
			const Vec3 globalUp (0, 0.2f, 0);

			// acceleration points toward the center of local path curvature, the
			// length determines how much the vehicle will roll while turning
			const Vec3 accelUp = smoothedAcceleration_ * 0.05f;

			// combined banking, sum of UP due to turning and global UP
			const Vec3 bankUp = accelUp + globalUp;

			// blend bankUp into vehicle's UP basis vector
			const float smoothRate = elapsedTime * 3;
			Vec3 tempUp = up();
			blendIntoAccumulator (smoothRate, bankUp, tempUp);
			setUp (tempUp.normalize());

			// adjust orthonormal basis vectors to be aligned with new velocity
			if (speed() > 0) regenerateOrthonormalBasisUF (newVelocity / speed());
		}

		/**
		 * Doing math stuff
		 */
		void regenerateOrthonormalBasisUF (const Vec3& newUnitForward) {
			forward_ = newUnitForward;

			// derive new side basis vector from NEW forward and OLD up
			side_.cross (forward_, up_);

			up_.cross (side_, forward_);
		}

		/**
		 * And another strang thing
		 */
		void measurePathCurvature (const float elapsedTime) {
			if (elapsedTime > 0) {
				const Vec3 dP = lastPosition_ - position ();
				const Vec3 dF = (lastForward_ - forward ()) / dP.length ();
				const Vec3 lateral = dF.perpendicularComponent (forward ());
				const float sign = (lateral.dot (side ()) < 0) ? 1.0f : -1.0f;
				curvature_ = lateral.length() * sign;
				blendIntoAccumulator (elapsedTime * 4.0f, curvature_, smoothedCurvature_);
				lastForward_  = forward ();
				lastPosition_ = position ();
			}
		}

		/**
		 * Got you
		 */
		void regenerateLocalSpaceForTerrainFollowing  (const Vec3& newVelocity) {

			const Vec3 surfaceNormal = position().normalize();
			const Vec3 surfacePoint = surfaceNormal * 40.0f;

			const Vec3 newUp = surfaceNormal;
			const Vec3 newPos = surfacePoint;
			const Vec3 newVel = newVelocity.perpendicularComponent(newUp);
			const float newSpeed = newVel.length();
			const Vec3 newFor = newVel / newSpeed;

			setSpeed (newSpeed);
			setPosition (newPos);
			setUp (newUp);
			setForward (newFor);
			side_.cross (forward_, up_);
		}

		/**
		 * Whatever
		 */
		Vec3 adjustRawSteeringForce (const Vec3& force) {
			static const float maxAdjustedSpeed = 0.2f * maxSpeed ();

			if ((speed () > maxAdjustedSpeed) || (force == Vec3::zero)) {
				return force;
			} else {
				const float range = speed() / maxAdjustedSpeed;
				const float cosine = interpolate (pow (range, 20), 1.0f, -1.0f);
				return limitMaxDeviationAngle (force, cosine, forward());
			}
		}

	private:
		Vec3 position_;
		Vec3 up_;
		Vec3 side_;
		Vec3 forward_;
		Vec3 smoothedAcceleration_;
		Vec3 smoothedPosition_;
		Vec3 lastForward_;
		Vec3 lastPosition_;

		float speed_;
		float curvature_;
		float smoothedCurvature_;

		static const float maxForce_;
		static const float maxSpeed_;
		static const float mass_;
		static const float radius_;
};

ObstacleGroup Boid::obstacles;
const float Boid::mass_     = 1.0f;
const float Boid::radius_   = 0.5f;
const float Boid::maxForce_ = 27.0f;
const float Boid::maxSpeed_ = 9.0f;


} // anonymous namespace

// specialice some OpenSteer functionality for our Boids class
namespace OpenSteer {
namespace Graphics {
template <>
class DrawVehicle<Boid> : public std::unary_function< const Boid&, void > {
	public:
		DrawVehicle( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder, Graphics::RenderFeeder::InstanceId _vehicleId ) : renderFeeder_( _renderFeeder ), vehicleId_( _vehicleId ) { }

		void operator() ( Boid const &_vehicle ) {
			renderFeeder_->render( localSpaceTransformationMatrix( _vehicle ), vehicleId_ );
		}

	private:
		SharedPointer< Graphics::RenderFeeder > renderFeeder_;
		Graphics::RenderFeeder::InstanceId vehicleId_;
}; // class DrawVehicle

}

/**
 * Specialization of @c OpenSteer::ProximityList::find_neighbours to work with
 * @c OpenSteer::AbstractVehicles.
 */
template< >
template< typename OutPutIterator >
void ProximityList< Boid* , Vec3>::find_neighbours( const Vec3 &position,  float const max_radius, OutPutIterator iter ) const {
	Boid* neighbours[neighbour_size_max];
	std::size_t neighbours_size = 0;

	{
		float const r2 = max_radius*max_radius;
		float max_neighbour_distance = 0.0f;
		std::size_t max_neighbour_distance_index = 0;
		
		const_iterator i=datastructure_.begin();
		
		while (i != datastructure_.end() && neighbours_size<neighbour_size_max) {
			Vec3 const offset = position - i->second;
			float const d2 = offset.lengthSquared();
			if (d2<r2) {
				if ( d2 > max_neighbour_distance ) {
					max_neighbour_distance = d2;
					max_neighbour_distance_index = neighbours_size;
				}
				neighbours[neighbours_size] = i->first;
				++neighbours_size;
			}
			++i;
		}

		while (i!=datastructure_.end()) {
			Vec3 const offset = position - i->second;
			float const d2 = offset.lengthSquared();
			if (d2<r2 && d2 < max_neighbour_distance ) {
				neighbours[ max_neighbour_distance_index ] = i->first;
				max_neighbour_distance = d2; // just temporary

				for ( std::size_t i = 0; i < neighbour_size_max; ++i ) {
					float const dist = ( position - neighbours[ i ]->position() ).lengthSquared();
					
					if ( dist > max_neighbour_distance ) {
						max_neighbour_distance = dist;
						max_neighbour_distance_index = i;
					}
				}
			}
			++i;
		}
	}

	if (neighbours_size < neighbour_size_max) {
		neighbours [neighbours_size] = 0;
		++neighbours_size;
	}

	for ( std::size_t i = 0; i < neighbours_size; ++i ) {
		*iter = neighbours[ i ];
		++iter;
	}
} // OpenSteer::ProximityList< Boid* , OpenSteer::Vec3>::find_neighbours


}


namespace {
class BoidOld : public OpenSteer::SimpleVehicle {
	public:
		// bknafla: separate update and simulation to adapt to parallelization.
		Vec3 simulate( float const /*currentTime*/, float const /*elapsedTime*/ ) { return Vec3::zero; }
		void update ( float const /*currentTime*/, float const /*elapsedTime*/, Vec3 const& /*_simulatedSteeringForce*/ ) { }
		void newPD (ProximityContainer* /*pc*/) { }
};



	// ----------------------------------------------------------------------------
	// PlugIn for OpenSteerDemo


class BoidsPlugIn : public PlugIn {
	private:
		// bknafla: Introduce parallelization. Replace proximity data structures.
		ProximityContainer pc_;
		
		// bknafla: adapt to thread safe random number generation.
		kapaga::random_number_source rand_source_;
		
		// bknafla: Change render interface to adapt to parallelization.
		Graphics::RenderFeeder::InstanceId boidGraphicsId_;
		
	public:

		const char* name (void) {return "BoidsRewrite";}
		float selectionOrderSortKey (void) {return 0.03f;}

		virtual ~BoidsPlugIn() {} // be more "nice" to avoid a compiler warning
		
		virtual void open ( SharedPointer< Graphics::RenderFeeder> const& _renderFeeder,
		                    SharedPointer< Graphics::GraphicsAnnotationLogger > const& _annotationLogger ) {
		
			// bknafla: Change render interface to adapt to parallelization.
			setAnnotationLogger( _annotationLogger );
			setRenderFeeder( _renderFeeder );

			// bknafla: Change render interface to adapt to parallelization.
			// Register boid graphical representation with render feeder.
			boidGraphicsId_ = 0;
			if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( *(Graphics::makeBasic3dSphericalVehicleGraphicsPrimitive( BoidOld( ), gGray70 ) ), boidGraphicsId_ ) ) {
				std::cerr << "Error: Unable to add graphical boid representation to render feeder." << std::endl;
			}
			
			// set up obstacles
			initObstacles ();
			
			
			// make the database used to accelerate proximity queries
			cyclePD = -1;
			nextPD ();
			
	
			// make default-sized flock
			OpenSteerDemo::selectedVehicle = NULL;
			population = 0;
			for ( std::size_t i = 0; i < gBoidStartCount ; ++i) {
				addBoidToFlock ();
			}

			simulatedSteeringForces.resize( flock.size(), Vec3::zero );
			
			
			// initialize camera
			OpenSteerDemo::init3dCamera (*OpenSteerDemo::selectedVehicle);
			OpenSteerDemo::camera.mode = Camera::cmFixed;
			OpenSteerDemo::camera.fixedDistDistance = OpenSteerDemo::cameraTargetDistance;
			OpenSteerDemo::camera.fixedDistVOffset = 0;
			OpenSteerDemo::camera.lookdownDistance = 20;
			OpenSteerDemo::camera.aimLeadTime = 0.5;
			OpenSteerDemo::camera.povOffset.set (0, 0.5, -2);
			
			reset_stop_watches();
		}

		std::vector< Vec3 > simulatedSteeringForces;
		
		virtual void update (const float /*currentTime*/, const float elapsedTime) {
			int const flock_size = flock.size();
			
			resume_simulation_stop_watch();
            // update flock simulation for each boid
            #pragma omp parallel for default( shared ) schedule(dynamic)
            for( int i = 0; i < flock_size; ++i ) {
                simulatedSteeringForces[ i ] = flock[ i ].simulate(pc_ );
                
            }
			suspend_simulation_stop_watch();
			

			resume_modification_stop_watch();
			resume_agent_update_stop_watch();
            //  schedule( static )
            #pragma omp parallel for default( shared ) schedule(static)
            for( int i = 0; i < flock_size; ++i ) {
                flock[ i ].update( elapsedTime, simulatedSteeringForces[ i ] );
                
            }
			suspend_agent_update_stop_watch();
			

			resume_spatial_data_structure_update_stop_watch();
            #pragma omp parallel for default( shared ) schedule(static)
            for( int i = 0; i < flock_size; ++i ) {
                pc_.update( &flock[ i ], flock[ i ].position() );
            }	
			suspend_spatial_data_structure_update_stop_watch();
			suspend_modification_stop_watch();
			
		}

		virtual void redraw (const float /*currentTime*/, const float /*elapsedTime*/) {
			
			std::for_each( flock.begin(), flock.end(),
			               OpenSteer::Graphics::DrawVehicle<Boid>( renderFeeder(), boidGraphicsId_ )
			             );
			
			// display status in the upper left corner of the window
			std::ostringstream status;
			status << "[F1/F2] " << population << " boids";
			status << "\n[F3]    PD type: ";
			switch (cyclePD) {
				case 0: status << "LQ bin lattice"; break;
				case 1: status << "brute force";    break;
			}
			status << "\n[F4]    Obstacles: ";
			switch (constraint) {
				case none:
					status << "none (wrap-around at sphere boundary)" ; break;
				case insideSphere:
					status << "inside a sphere" ; break;
				case outsideSphere:
					status << "inside a sphere, outside another" ; break;
				case outsideSpheres:
					status << "inside a sphere, outside several" ; break;
				case outsideSpheresNoBig:
					status << "outside several spheres, with wrap-around" ; break;
				case rectangle:
					status << "inside a sphere, with a rectangle" ; break;
				case rectangleNoBig:
					status << "a rectangle, with wrap-around" ; break;
				case outsideBox:
					status << "inside a sphere, outside a box" ; break;
				case insideBox:
					status << "inside a box" ; break;
			}
			status << std::endl;
			
			const Vec3 screenLocation (10.0f, 50.0f, 0.0f );
			
			renderFeeder()->render( Graphics::TextAt2dLocationGraphicsPrimitive( screenLocation, Graphics::TextAt2dLocationGraphicsPrimitive::TOP_LEFT, status.str(), gGray80 ) );

            drawObstacles ();
        }

		virtual void close (void) {
			// delete each member of the flock
			while (population > 0) {
				removeBoidFromFlock ();
			}
		}

		virtual void reset (void) {
			// bknafla: adapt to thread safe random number generation.
			// reset each boid in flock
			for (iterator i = flock.begin(); i != flock.end(); ++i ) {
				// Boids determine their position using a random number source. They copy
				// the random number source of the plugin which is changed (a random number is 
				// drawn for every boid) so each boid gets a different first random number from it.
				rand_source_.draw();
				(*i).reset( rand_source_ );
				
				// bknafla: Introduce parallelization. Replace proximity data structures.
				pc_.update( &(*i), (*i).position() );
			}

			// reset camera position
			OpenSteerDemo::position3dCamera (*OpenSteerDemo::selectedVehicle);

            // make camera jump immediately to new position
            OpenSteerDemo::camera.doNotSmoothNextMove ();
			
			// Reset the stop watches that measure detailed time info in the update method.
			reset_stop_watches();
        }
		
		// for purposes of demonstration, allow cycling through various
		// types of proximity databases.  this routine is called when the
		// OpenSteerDemo user pushes a function key.
		void nextPD (void) {
		}

		void handleFunctionKeys (int keyNumber) {
			switch (keyNumber) {
				case 1:
					for ( std::size_t i = 0; i < add_vehicle_count; ++i ) {
						addBoidToFlock ();
					}
					simulatedSteeringForces.resize( flock.size(), Vec3::zero );
					break;
				case 2:
					for ( std::size_t i = 0; i < add_vehicle_count; ++i ) {
						removeBoidFromFlock ();
					}
					simulatedSteeringForces.resize( flock.size(), Vec3::zero );
					break;
				case 3:  /*nextPD ();*/                 break;
				case 4:  nextBoundaryCondition ();  break;
				case 5:  break;
				default: break;
			}
        	}

		void printLQbinStats (void) { }

		void printMiniHelpForFunctionKeys (void) {
			std::ostringstream message;
			message << "Function keys handled by ";
			message << '"' << name() << '"' << ':' << std::ends;
			OpenSteerDemo::printMessage (message);
			OpenSteerDemo::printMessage ("  F1     add a boid to the flock.");
			OpenSteerDemo::printMessage ("  F2     remove a boid from the flock.");
			OpenSteerDemo::printMessage ("  F3     use next proximity database.");
			OpenSteerDemo::printMessage ("  F4     next flock boundary condition.");
			OpenSteerDemo::printMessage ("");
		}

		void addBoidToFlock (void) {
			// bknafla: adapt to thread safe random number generation.
			// Boids determine their position using a random number source. They copy
			// the random number source of the plugin which is changed (a random number is 
			// drawn for every boid) so each boid gets a different first random number from it.
			rand_source_.draw();
			flock.push_back (Boid (rand_source_) );
			++population;
            
			// bknafla: Introduce parallelization. Replace proximity data structures.
			pc_.add( &flock.back(), flock.back().position() );
			
        	}

		void removeBoidFromFlock (void) {
			if (population > 0) {
				// bknafla: Introduce parallelization. Replace proximity data structures.
				// Had to remove @c const from pointer.
				// save a pointer to the last boid, then remove it from the flock
				flock.pop_back();
				--population;

				// bknafla: Introduce parallelization. Replace proximity data structures.
				pc_.remove( &flock.back() );
			}
		}
		
		// bknafla: Change render interface to adapt to parallelization.
		virtual void setRenderFeeder( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder ) {
			PlugIn::setRenderFeeder( _renderFeeder );
			
			//std::for_each( flock.begin(), flock.end(), OpenSteer::Graphics::SetRenderFeeder< Boid >( _renderFeeder ) );
		}
		
		// bknafla: Change render interface to adapt to parallelization.
		virtual void setAnnotationLogger( SharedPointer< Graphics::GraphicsAnnotationLogger > const& _annotationLogger ) {
			PlugIn::setAnnotationLogger( _annotationLogger );
			
			//std::for_each( flock.begin(), flock.end(), OpenSteer::Graphics::SetAnnotationLogger< Boid >( _annotationLogger ) );
		}
		

		// return an AVGroup containing each boid of the flock
		const AVGroup& allVehicles (void) {static const AVGroup temp; return temp;}

		// flock: a group (STL vector) of pointers to all boids
		Boid::groupType flock;
		typedef Boid::groupType::iterator iterator;


		// keep track of current flock size
		int population;

		// which of the various proximity databases is currently in use
		int cyclePD;

        // --------------------------------------------------------
        // the rest of this plug-in supports the various obstacles:
        // --------------------------------------------------------

        // enumerate demos of various constraints on the flock
        enum ConstraintType {none, insideSphere,
                             outsideSphere, outsideSpheres, outsideSpheresNoBig,
                             rectangle, rectangleNoBig,
                             outsideBox, insideBox};

        ConstraintType constraint;

        // select next "boundary condition / constraint / obstacle"
        void nextBoundaryCondition (void)
        {
			// @attention bknafla: Is it standard conform to rely on an overflow in the enum type??
            constraint = (ConstraintType) ((int) constraint + 1);
            updateObstacles ();
        }

        class SO : public SphereObstacle
        {
			void draw ( const bool /* filled */, 
						const Color& color, 
						// const Vec3& vp,
						OpenSteer::Graphics::RenderFeeder& rf ) const
            {
				// drawSphereObstacle (*this, 10.0f, filled, color, vp);
				
				rf.render( OpenSteer::translationMatrix( center ), 
						   // Matrix( Vec3( -1.0f, 0.0f, 0.0f ),
						   //	      Vec3( 0.0f, -1.0f, 0.0f ),
						   //         Vec3( 0.0f, 0.0f, 1.0f ),
						   //		  center ),
						   OpenSteer::Graphics::SphereGraphicsPrimitive( radius, color, 10, 10 ) );
				
			}
		};

        class RO : public RectangleObstacle
        {
			void draw (const bool /* filled */, 
					   const Color& color, 
					   // const Vec3& position,
					   OpenSteer::Graphics::RenderFeeder& rf ) const
            {
				//rf.render( OpenSteer::localSpaceTransformationMatrix( *this ),
				//		   OpenSteer::Graphics::BoxGraphicsPrimitive( width, height, depth, color ) );
				tempDrawRectangle( *this, color, rf );
			}
		};

        class BO : public BoxObstacle
        {
			void draw (const bool /* filled */, 
					   const Color& color, 
					   // const Vec3& position,
					   OpenSteer::Graphics::RenderFeeder& rf ) const
            {
				// The @c -side() is needed because the local coordinate space of the obstacle
				// and that of the box graphics primitive are a bit different.
				// The way forward and side are used once for the local space coordinate system
				// and then for the vehicles completely drives me crazy...
				rf.render( Matrix( -side(), up(), forward(), position() ),
						   OpenSteer::Graphics::BoxGraphicsPrimitive( width, height, depth, color ) );
				// tempDrawBox (*this, color);
			}
		};

		RO bigRectangle;
		BO outsideBigBox, insideBigBox;
		SO insideBigSphere, outsideSphere0, outsideSphere1, outsideSphere2, outsideSphere3, outsideSphere4, outsideSphere5, outsideSphere6;


		void initObstacles (void) {
			constraint = none;

			insideBigSphere.radius = worldRadius;
			insideBigSphere.setSeenFrom (Obstacle::inside);

			outsideSphere0.radius = worldRadius * 0.5f;

			const float r = worldRadius * 0.33f;
			outsideSphere1.radius = r;
			outsideSphere2.radius = r;
			outsideSphere3.radius = r;
			outsideSphere4.radius = r;
			outsideSphere5.radius = r;
			outsideSphere6.radius = r;

			const float p = worldRadius * 0.5f;
			const float m = -p;
			const float z = 0.0f;
			outsideSphere1.center.set (p, z, z);
			outsideSphere2.center.set (m, z, z);
			outsideSphere3.center.set (z, p, z);
			outsideSphere4.center.set (z, m, z);
			outsideSphere5.center.set (z, z, p);
			outsideSphere6.center.set (z, z, m);

			const Vec3 tiltF = Vec3 (1.0f, 1.0f, 0.0f).normalize ();
			const Vec3 tiltS (0.0f, 0.0f, 1.0f);
			const Vec3 tiltU = Vec3 (-1.0f, 1.0f, 0.0f).normalize ();

			bigRectangle.width = 50.0f;
			bigRectangle.height = 80.0f;
			bigRectangle.setSeenFrom (Obstacle::both);
			bigRectangle.setForward (tiltF);
			bigRectangle.setSide (tiltS);
			bigRectangle.setUp (tiltU);

			outsideBigBox.width = 50.0f;
			outsideBigBox.height = 80.0f;
			outsideBigBox.depth = 20.0f;
			outsideBigBox.setForward (tiltF);
			outsideBigBox.setSide (tiltS);
			outsideBigBox.setUp (tiltU);

			insideBigBox = outsideBigBox;
			insideBigBox.setSeenFrom (Obstacle::inside);

			updateObstacles ();
		}


        // update Boid::obstacles list when constraint changes
        void updateObstacles (void)
        {
            // first clear out obstacle list
            Boid::obstacles.clear ();

            // add back obstacles based on mode
            switch (constraint)
            {
            default:
                // reset for wrap-around, fall through to first case:
                constraint = none;
            case none:
                break;
            case insideSphere:
                Boid::obstacles.push_back (&insideBigSphere);
                break;
            case outsideSphere:
                Boid::obstacles.push_back (&insideBigSphere);
                Boid::obstacles.push_back (&outsideSphere0);
                break;
            case outsideSpheres:
                Boid::obstacles.push_back (&insideBigSphere);
            case outsideSpheresNoBig:
                Boid::obstacles.push_back (&outsideSphere1);
                Boid::obstacles.push_back (&outsideSphere2);
                Boid::obstacles.push_back (&outsideSphere3);
                Boid::obstacles.push_back (&outsideSphere4);
                Boid::obstacles.push_back (&outsideSphere5);
                Boid::obstacles.push_back (&outsideSphere6);
                break;
            case rectangle:
                Boid::obstacles.push_back (&insideBigSphere);
                Boid::obstacles.push_back (&bigRectangle);
            case rectangleNoBig:
                Boid::obstacles.push_back (&bigRectangle);
                break;
            case outsideBox:
                Boid::obstacles.push_back (&insideBigSphere);
                Boid::obstacles.push_back (&outsideBigBox);
                break;
            case insideBox:
                Boid::obstacles.push_back (&insideBigBox);
                break;
            }
        }


        void drawObstacles (void)
        {
            for (ObstacleIterator o = Boid::obstacles.begin();
                 o != Boid::obstacles.end();
                 ++o)
            {
                (**o).draw ( false, // draw in wireframe
                             ( ( *o == &insideBigSphere ) ?
							   Color (0.2f, 0.2f, 0.4f) :
							   Color (0.1f, 0.1f, 0.2f)),
							// OpenSteerDemo::camera.position (),
							*renderFeeder() );
				
            }
        }


        static void tempDrawRectangle (const RectangleObstacle& rect, 
									   const Color& color, 
									   OpenSteer::Graphics::RenderFeeder& rf )
        {
            float const w = rect.width / 2;
            float const h = rect.height / 2;

            Vec3 const v1 = rect.globalizePosition (Vec3 ( w,  h, 0));
            Vec3 const v2 = rect.globalizePosition (Vec3 (-w,  h, 0));
            Vec3 const v3 = rect.globalizePosition (Vec3 (-w, -h, 0));
            Vec3 const v4 = rect.globalizePosition (Vec3 ( w, -h, 0));

            rf.render( OpenSteer::Graphics::LineGraphicsPrimitive( v1, v2, color ) );
            rf.render( OpenSteer::Graphics::LineGraphicsPrimitive( v2, v3, color ) );
            rf.render( OpenSteer::Graphics::LineGraphicsPrimitive( v3, v4, color ) );
            rf.render( OpenSteer::Graphics::LineGraphicsPrimitive( v4, v1, color ) );
        }
    };


BoidsPlugIn gBoidsPlugIn;



    // ----------------------------------------------------------------------------

} // anonymous namespace
