#if 0
outcommented because it segfaults, when used in conjunction with other plugins

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

#include "OpenSteer/Matrix.h"

// include the cupp files
#include "cupp/vector.h"
#include "cupp/kernel.h"

// CUDA
#include <vector_types.h>

#include "OpenSteer/kernels.h"
#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/Boid.h"

#include "ds/gpu_grid.h"
#include "OpenSteer/PosIndexPair.h"

// Include names declared in the OpenSteer namespace into the
// namespaces to search to find names.
using namespace OpenSteer;

namespace B7_tf{



ObstacleGroup obstacles;

} // anonymous namespace

using namespace B7_tf;

namespace B7_tf {
class BoidOld : public OpenSteer::SimpleVehicle {
	public:
		// bknafla: separate update and simulation to adapt to parallelization.
		Vec3 simulate( float const /*currentTime*/, float const /*elapsedTime*/ ) { return Vec3::zero; }
		void update ( float const /*currentTime*/, float const /*elapsedTime*/, Vec3 const& /*_simulatedSteeringForce*/ ) { }
		void newPD (...) { }
};

;





	// ----------------------------------------------------------------------------
	// PlugIn for OpenSteerDemo

class BoidsPlugIn : public PlugIn {
	private:
		// bknafla: adapt to thread safe random number generation.
		kapaga::random_number_source rand_source_;
		
		// bknafla: Change render interface to adapt to parallelization.
		Graphics::RenderFeeder::InstanceId boidGraphicsId_[3];

		// the new cupp "proximity datastructure"
		cupp::vector< Vec3 >   positions_;
		cupp::vector< Vec3 >   forwards_;
		cupp::vector< Vec3 >   steering_result_;
		cupp::vector< Boid >   flock;
		cupp::vector< Matrix > render_position_;

		cupp::kernel find_neighbours;
		cupp::kernel update_flock;

		cupp::kernel count;
		cupp::kernel prescan;
		cupp::kernel fill;

		ds::gpu_grid grid_;

		int think_cycle_;
	public:

		const char* name (void) {return "BoidsCuPP inkl. simulate inkl. update with thinking frequency with GPU-grid V1+";}
		float selectionOrderSortKey (void) {return 0.03f;}

		BoidsPlugIn() :
		find_neighbours(get_find_neighbours_simulate_frequency_gpu_grid_kernel(), dim3 (threads_per_block/threads_per_block), dim3 (threads_per_block)),
		update_flock (get_update_kernel(), dim3 (threads_per_block/threads_per_block), dim3 (threads_per_block)),
		count (get_v1_count_kernel(), dim3 (grid_nb_of_cells/threads_per_block), dim3 (threads_per_block)),
		prescan (get_v1_prescan_kernel(), dim3 (1), dim3 (/*512*/grid_nb_of_cells/2)),
		fill (get_v1_fill_kernel(), dim3 (grid_nb_of_cells/threads_per_block), dim3 (threads_per_block)),
		grid_(boid_worldRadius, grid_size)
		{}

		virtual ~BoidsPlugIn() {} // be more "nice" to avoid a compiler warning
		
		virtual void open ( SharedPointer< Graphics::RenderFeeder> const& _renderFeeder,
		                    SharedPointer< Graphics::GraphicsAnnotationLogger > const& _annotationLogger ) {
		
			// bknafla: Change render interface to adapt to parallelization.
			setAnnotationLogger( _annotationLogger );
			setRenderFeeder( _renderFeeder );

			// bknafla: Change render interface to adapt to parallelization.
			// Register boid graphical representation with render feeder.
			for (int i=0; i<3; ++i) {
				boidGraphicsId_[i] = i;
			}
			
			if ( ! renderFeeder()->addToGraphicsPrimitiveLibrary( *(Graphics::makeBasic3dSphericalVehicleGraphicsPrimitive( BoidOld( ), gOrange ) ), boidGraphicsId_[0] ) ) {
				std::cerr << "Error: Unable to add graphical boid representation to render feeder." << std::endl;
			}
			renderFeeder()->addToGraphicsPrimitiveLibrary( *(Graphics::makeBasic3dSphericalVehicleGraphicsPrimitive( BoidOld( ), gMagenta ) ), boidGraphicsId_[1] );

			renderFeeder()->addToGraphicsPrimitiveLibrary( *(Graphics::makeBasic3dSphericalVehicleGraphicsPrimitive( BoidOld( ), gYellow ) ), boidGraphicsId_[2] );

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
			
			think_cycle_ = 0;
			
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
			
			// bknafla: Determine which boids to simulate in this frame.
			think_cycle_ = ( think_cycle_ + 1 ) % think_frequency;
			
			int const thinking_boid_count_per_frame = static_cast< int >( std::ceil( static_cast< float >( flock_size ) / static_cast< float >( think_frequency ) ) );
			int const think_lower_boid_index = think_cycle_ * thinking_boid_count_per_frame;


			if (flock_size != steering_result_.size()) {
				steering_result_.resize(flock_size);
				render_position_.resize(flock_size, Matrix(Vec3(), Vec3(0, 1, 0), Vec3(), Vec3()) );
			
				dim3 block_dim(threads_per_block);
				
				find_neighbours.set_block_dim (block_dim);
				find_neighbours.set_grid_dim (thinking_boid_count_per_frame/threads_per_block);
				
				update_flock.set_block_dim (block_dim);
				update_flock.set_grid_dim (flock_size/block_dim.x);

				grid_.resize(flock_size);
			}

			count   (cupp_device, positions_, grid_, flock_size);
			prescan (cupp_device, grid_, flock_size);
			fill    (cupp_device, positions_, grid_, flock_size);

			find_neighbours (cupp_device, think_lower_boid_index, grid_, positions_, forwards_, steering_result_);

			update_flock(cupp_device, elapsedTime, positions_, forwards_, steering_result_, flock, render_position_);

			cupp_device.sync();
			
			suspend_simulation_stop_watch();
		}

		virtual void redraw (const float /*currentTime*/, const float /*elapsedTime*/) {

			for (std::size_t i=0; i<render_position_.size(); ++i) {
				renderFeeder() -> render (render_position_[i], boidGraphicsId_[i%3]);
			}
			
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
				forwards_[i] = RandomUnitVector ( rand_source_ );
				flock[i].get().reset( rand_source_, forwards_[i] );

				positions_[i] = RandomVectorInUnitRadiusSphere( rand_source_ ) * 20.0f;
				
			}
		}
		
		// for purposes of demonstration, allow cycling through various
		// types of proximity databases.  this routine is called when the
		// OpenSteerDemo user pushes a function key.
		void nextPD (void) { }

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
			rand_source_.draw();
			
			forwards_.push_back( RandomUnitVector ( rand_source_ ) );
			flock.push_back (Boid (rand_source_, forwards_.back()) );
			
			positions_.push_back( RandomVectorInUnitRadiusSphere( rand_source_ ) * 20.0f );

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


        // update obstacles list when constraint changes
        void updateObstacles (void)
        {
            // first clear out obstacle list
            obstacles.clear ();

            // add back obstacles based on mode
            switch (constraint)
            {
            default:
                // reset for wrap-around, fall through to first case:
                constraint = none;
            case none:
                break;
            case insideSphere:
                obstacles.push_back (&insideBigSphere);
                break;
            case outsideSphere:
                obstacles.push_back (&insideBigSphere);
                obstacles.push_back (&outsideSphere0);
                break;
            case outsideSpheres:
                obstacles.push_back (&insideBigSphere);
            case outsideSpheresNoBig:
                obstacles.push_back (&outsideSphere1);
                obstacles.push_back (&outsideSphere2);
                obstacles.push_back (&outsideSphere3);
                obstacles.push_back (&outsideSphere4);
                obstacles.push_back (&outsideSphere5);
                obstacles.push_back (&outsideSphere6);
                break;
            case rectangle:
                obstacles.push_back (&insideBigSphere);
                obstacles.push_back (&bigRectangle);
            case rectangleNoBig:
                obstacles.push_back (&bigRectangle);
                break;
            case outsideBox:
                obstacles.push_back (&insideBigSphere);
                obstacles.push_back (&outsideBigBox);
                break;
            case insideBox:
                obstacles.push_back (&insideBigBox);
                break;
            }
        }


        void drawObstacles (void)
        {
            for (ObstacleIterator o = obstacles.begin();
                 o != obstacles.end();
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

#endif
