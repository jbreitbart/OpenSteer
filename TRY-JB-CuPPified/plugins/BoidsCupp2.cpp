#include <iostream>
#include <limits>
#include <cstddef>
#include <sstream>
#include "kapaga/random_number_source.h"
#include "OpenSteer/SimpleVehicle.h"
#include "OpenSteer/OpenSteerDemo.h"
#include "OpenSteer/ProximityList.h"
#include "OpenSteer/Color.h"
#include "OpenSteer/Graphics/PlugInRenderingUtilities.h"
#include "OpenSteer/PlugInUtilities.h"
#include "OpenSteer/Graphics/GraphicsPrimitivesUtilities.h"

// Include OpenSteer::localSpaceTransformationMatrix, OpenSteer::translationMatrix
#include "OpenSteer/MatrixUtilities.h"


// include the cupp files
#include "cupp/vector.h"
#include "cupp/kernel.h"

// CUDA
#include <vector_types.h>

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

// Include names declared in the OpenSteer namespace into the
// namespaces to search to find names.
using namespace OpenSteer;

namespace B2{


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
			speed_ = boid_maxSpeed * 0.3f;

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
		float radius  () const  { return boid_radius;       }
		float maxForce() const  { return boid_maxForce;     }
		float maxSpeed() const  { return boid_maxSpeed;     }
		float mass    () const  { return boid_mass;         }
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
			if (position().length() > boid_worldRadius) {
				// wrap around (teleport)
				setPosition (position().sphericalWrapAround (Vec3::zero, boid_worldRadius));
			}
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
};

ObstacleGroup Boid::obstacles;

} // anonymous namespace

using namespace B2;
// specialice some OpenSteer functionality for our Boids class
namespace OpenSteer {
namespace Graphics {
template <>
class DrawVehicle<B2::Boid> : public std::unary_function< const B2::Boid&, void > {
	public:
		DrawVehicle( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder, Graphics::RenderFeeder::InstanceId _vehicleId ) : renderFeeder_( _renderFeeder ), vehicleId_( _vehicleId ) { }

		void operator() ( B2::Boid const &_vehicle ) {
			renderFeeder_->render( localSpaceTransformationMatrix( _vehicle ), vehicleId_ );
		}

	private:
		SharedPointer< Graphics::RenderFeeder > renderFeeder_;
		Graphics::RenderFeeder::InstanceId vehicleId_;
}; // class DrawVehicle

}


}


namespace B2 {
class BoidOld : public OpenSteer::SimpleVehicle {
	public:
		// bknafla: separate update and simulation to adapt to parallelization.
		Vec3 simulate( float const /*currentTime*/, float const /*elapsedTime*/ ) { return Vec3::zero; }
		void update ( float const /*currentTime*/, float const /*elapsedTime*/, Vec3 const& /*_simulatedSteeringForce*/ ) { }
		void newPD (...) { }
};



	// ----------------------------------------------------------------------------
	// PlugIn for OpenSteerDemo

class BoidsPlugIn : public PlugIn {
	private:
		// bknafla: Introduce parallelization. Replace proximity data structures.
		//ProximityContainer pc_;
		
		// bknafla: adapt to thread safe random number generation.
		kapaga::random_number_source rand_source_;
		
		// bknafla: Change render interface to adapt to parallelization.
		Graphics::RenderFeeder::InstanceId boidGraphicsId_;

		// the new cupp "proximity datastructure"
		cupp::vector< Vec3  >  positions_;
		cupp::vector< Vec3  >  forwards_;
		cupp::vector< Vec3 >   steering_result_;

		
	public:

		const char* name (void) {return "BoidsCuPP inkl. simulate";}
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

		
		virtual void update (const float /*currentTime*/, const float elapsedTime) {
			// bknafla: separate update and simulation to adapt to parallelization.
			unsigned int const flock_size = flock.size();

			if (flock_size==0) return;

            resume_simulation_stop_watch();
            
			if (flock_size != steering_result_.size()) {
				steering_result_.resize(flock_size);
			}
			
			dim3 block_dim(threads_per_block);
			dim3 grid_dim(flock_size/block_dim.x);
			cupp::kernel find_neighbours(get_find_neighbours_simulate_kernel(), grid_dim, block_dim);

			find_neighbours (cupp_device, positions_, forwards_, steering_result_);

			// get the results form the device
			// this should NOT be done in parallel region on the CPU
			// as cupp::vector is NOT thread safe
			steering_result_.update_host();
			
			suspend_simulation_stop_watch();


            resume_modification_stop_watch();
            #pragma omp parallel for default( shared ) schedule(static)
            for( std::size_t i = 0; i < flock_size; ++i ) {
                //std::cout << steering_result_[ i ] << std::endl;
                flock[ i ].update( elapsedTime, steering_result_[ i ] );
                
                positions_[i] = flock[i].position();
                forwards_[i]  = flock[i].forward();
            }
            suspend_modification_stop_watch();

		}

		virtual void redraw (const float /*currentTime*/, const float /*elapsedTime*/) {
			
			std::for_each( flock.begin(), flock.end(),
			               OpenSteer::Graphics::DrawVehicle<B2::Boid>( renderFeeder(), boidGraphicsId_ )
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
			for (std::size_t i = 0; i<flock.size(); ++i ) {
				// Boids determine their position using a random number source. They copy
				// the random number source of the plugin which is changed (a random number is 
				// drawn for every boid) so each boid gets a different first random number from it.
				rand_source_.draw();
				flock[i].reset( rand_source_ );
				
				positions_[i] = flock[i].position();
				forwards_[i] = flock[i].forward();
			}

			// reset camera position
			//OpenSteerDemo::position3dCamera (*OpenSteerDemo::selectedVehicle);

			// make camera jump immediately to new position
			//OpenSteerDemo::camera.doNotSmoothNextMove ();
			
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
					break;
				case 2:
					for ( std::size_t i = 0; i < add_vehicle_count; ++i ) {
						removeBoidFromFlock ();
					}
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
			flock.push_back (B2::Boid (rand_source_) );
			
			positions_.push_back( flock.back().position() );
			forwards_.push_back( flock.back().forward() );
			//references_.push_back (population);

			++population;

        	}

		void removeBoidFromFlock (void) {
			if (population > 0) {
				// bknafla: Introduce parallelization. Replace proximity data structures.
				// Had to remove @c const from pointer.
				// save a pointer to the last boid, then remove it from the flock
				flock.pop_back();
				--population;

				// bknafla: Introduce parallelization. Replace proximity data structures.
				positions_.pop_back();
				forwards_.pop_back();
				//references_.pop_back();
			}
		}
		
		// bknafla: Change render interface to adapt to parallelization.
		virtual void setRenderFeeder( SharedPointer< Graphics::RenderFeeder > const& _renderFeeder ) {
			PlugIn::setRenderFeeder( _renderFeeder );
		}
		
		// bknafla: Change render interface to adapt to parallelization.
		virtual void setAnnotationLogger( SharedPointer< Graphics::GraphicsAnnotationLogger > const& _annotationLogger ) {
			PlugIn::setAnnotationLogger( _annotationLogger );
		}
		

		// return an AVGroup containing each boid of the flock
		const AVGroup& allVehicles (void) {static const AVGroup temp; return temp;}

		// flock: a group (STL vector) of pointers to all boids
		B2::Boid::groupType flock;
		typedef B2::Boid::groupType::iterator iterator;


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

			insideBigSphere.radius = boid_worldRadius;
			insideBigSphere.setSeenFrom (Obstacle::inside);

			outsideSphere0.radius = boid_worldRadius * 0.5f;

			const float r = boid_worldRadius * 0.33f;
			outsideSphere1.radius = r;
			outsideSphere2.radius = r;
			outsideSphere3.radius = r;
			outsideSphere4.radius = r;
			outsideSphere5.radius = r;
			outsideSphere6.radius = r;

			const float p = boid_worldRadius * 0.5f;
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
            B2::Boid::obstacles.clear ();

            // add back obstacles based on mode
            switch (constraint)
            {
            default:
                // reset for wrap-around, fall through to first case:
                constraint = none;
            case none:
                break;
            case insideSphere:
                B2::Boid::obstacles.push_back (&insideBigSphere);
                break;
            case outsideSphere:
                B2::Boid::obstacles.push_back (&insideBigSphere);
                B2::Boid::obstacles.push_back (&outsideSphere0);
                break;
            case outsideSpheres:
                B2::Boid::obstacles.push_back (&insideBigSphere);
            case outsideSpheresNoBig:
                B2::Boid::obstacles.push_back (&outsideSphere1);
                B2::Boid::obstacles.push_back (&outsideSphere2);
                B2::Boid::obstacles.push_back (&outsideSphere3);
                B2::Boid::obstacles.push_back (&outsideSphere4);
                B2::Boid::obstacles.push_back (&outsideSphere5);
                B2::Boid::obstacles.push_back (&outsideSphere6);
                break;
            case rectangle:
                B2::Boid::obstacles.push_back (&insideBigSphere);
                B2::Boid::obstacles.push_back (&bigRectangle);
            case rectangleNoBig:
                B2::Boid::obstacles.push_back (&bigRectangle);
                break;
            case outsideBox:
                B2::Boid::obstacles.push_back (&insideBigSphere);
                B2::Boid::obstacles.push_back (&outsideBigBox);
                break;
            case insideBox:
                B2::Boid::obstacles.push_back (&insideBigBox);
                break;
            }
        }


        void drawObstacles (void)
        {
            for (ObstacleIterator o = B2::Boid::obstacles.begin();
                 o != B2::Boid::obstacles.end();
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
