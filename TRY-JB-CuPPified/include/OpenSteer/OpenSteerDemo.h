

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
// OpenSteerDemo
//
// This class encapsulates the state of the OpenSteerDemo application and
// the services it provides to its plug-ins
//
// 10-04-04 bk:  put everything into the OpenSteer namespace
// 11-14-02 cwr: recast App class as OpenSteerDemo 
// 06-26-02 cwr: App class created 
//
//
// ----------------------------------------------------------------------------


#ifndef OPENSTEER_OPENSTEERDEMO_H
#define OPENSTEER_OPENSTEERDEMO_H



// Include std::size_t
#include <cstddef>

// Include std::ostringstream
#include <sstream>

// Include kapaga::omp_stop_watch, kapaga::omp_stop_watch::time_type
#include "kapaga/omp_stop_watch.h"

#include "OpenSteer/Clock.h"
#include "OpenSteer/PlugIn.h"
#include "OpenSteer/Camera.h"
#include "OpenSteer/Utilities.h"

// Include OpenSteer::Graphics::OpenGlRenderService. OpenSteer::Graphics::OpenGlRenderer, OpenSteer::Graphics::OpenGlRenderFeeder
#include "OpenSteer/Graphics/OpenGlRenderService.h"

// Include OpenSteer::SharedPointer
#include "OpenSteer/SharedPointer.h"

// Include OpenSteer::Graphics::BatchingRenderFeeder
// #include "OpenSteer/Graphics/BatchingRenderFeeder.h"

// Include OpenSteer::Graphics::LockingRenderFeeder
// #include "OpenSteer/Graphics/LockingRenderFeeder.h"

// Include OpenSteer::ThreadStorage
#include "kapaga/thread_storage.h"

// Include OpenSteer::Graphics::ThreadStorageRenderFeeder
#include "OpenSteer/Graphics/ThreadStorageRenderFeeder.h"

// Include omp_get_num_procs, omp_get_max_threads
#include "kapaga/omp_header_wrapper.h"


namespace OpenSteer {

    class Color;
    class Vec3;
    

    class OpenSteerDemo
    {
    public:
        // ------------------------------------------------------ component objects

        // clock keeps track of both "real time" and "simulation time"
        static Clock clock;

        // camera automatically tracks selected vehicle
        static Camera camera;

        // ------------------------------------------ addresses of selected objects

        // currently selected plug-in (user can choose or cycle through them)
        static PlugIn* selectedPlugIn;

        // currently selected vehicle.  Generally the one the camera follows and
        // for which additional information may be displayed.  Clicking the mouse
        // near a vehicle causes it to become the Selected Vehicle.
        static AbstractVehicle* selectedVehicle;

        // -------------------------------------------- initialize, update and exit

        // initialize OpenSteerDemo
        //     XXX  if I switch from "totally static" to "singleton"
        //     XXX  class structure this becomes the constructor
        static void initialize (void);

        // main update function: step simulation forward and redraw scene
        static void updateSimulationAndRedraw (void);

        // exit OpenSteerDemo with a given text message or error code
        static void errorExit (const char* message);
        static void exit (int exitCode);

        // ------------------------------------------------------- PlugIn interface

        // select the default PlugIn
        static void selectDefaultPlugIn (void);
        
        // select the "next" plug-in, cycling through "plug-in selection order"
        static void selectNextPlugIn (void);

        // handle function keys an a per-plug-in basis
        static void functionKeyForPlugIn (int keyNumber);

        // return name of currently selected plug-in
        static const char* nameOfSelectedPlugIn (void);

        // open the currently selected plug-in
        static void openSelectedPlugIn (void);

        // do a simulation update for the currently selected plug-in
        static void updateSelectedPlugIn (const float currentTime,
                                          const float elapsedTime);

        // redraw graphics for the currently selected plug-in
        static void redrawSelectedPlugIn (const float currentTime,
                                          const float elapsedTime);

        // close the currently selected plug-in
        static void closeSelectedPlugIn (void);

        // reset the currently selected plug-in
        static void resetSelectedPlugIn (void);

        static const AVGroup& allVehiclesOfSelectedPlugIn(void);

        // ---------------------------------------------------- OpenSteerDemo phase

        static bool phaseIsDraw     (void) {return phase == drawPhase;}
        static bool phaseIsUpdate   (void) {return phase == updatePhase;}
        static bool phaseIsOverhead (void) {return phase == overheadPhase;}

        static float phaseTimerDraw     (void) {return phaseTimers[drawPhase];}
        static float phaseTimerUpdate   (void) {return phaseTimers[updatePhase];}
        // XXX get around shortcomings in current implementation, see note
        // XXX in updateSimulationAndRedraw
        //static float phaseTimerOverhead(void){return phaseTimers[overheadPhase];}
        static float phaseTimerOverhead (void)
        {
            return (clock.getElapsedRealTime() -
                    (phaseTimerDraw() + phaseTimerUpdate()));
        }

        // ------------------------------------------------------ delayed reset XXX

        // XXX to be reconsidered
        static void queueDelayedResetPlugInXXX (void);
        static void doDelayedResetPlugInXXX (void);

        // ------------------------------------------------------ vehicle selection

        // select the "next" vehicle: cycle through the registry
        static void selectNextVehicle (void);

        // select vehicle nearest the given screen position (e.g.: of the mouse)
        static void selectVehicleNearestScreenPosition (int x, int y);

        // ---------------------------------------------------------- mouse support

        // Find the AbstractVehicle whose screen position is nearest the
        // current the mouse position.  Returns NULL if mouse is outside
        // this window or if there are no AbstractVehicles.
        static AbstractVehicle* vehicleNearestToMouse (void);

        // Find the AbstractVehicle whose screen position is nearest the
        // given window coordinates, typically the mouse position.  Note
        // this will return NULL if there are no AbstractVehicles.
        static AbstractVehicle* findVehicleNearestScreenPosition (int x, int y);

        // for storing most recent mouse state
        static int mouseX;
        static int mouseY;
        static bool mouseInWindow;

        // ------------------------------------------------------- camera utilities

        // set a certain initial camera state used by several plug-ins
        static void init2dCamera (AbstractVehicle& selected);
        static void init2dCamera (AbstractVehicle& selected,
                                  float distance,
                                  float elevation);
        static void init3dCamera (AbstractVehicle& selected);
        static void init3dCamera (AbstractVehicle& selected,
                                  float distance,
                                  float elevation);

        // set initial position of camera based on a vehicle
        static void position3dCamera (AbstractVehicle& selected);
        static void position3dCamera (AbstractVehicle& selected,
                                      float distance,
                                      float elevation);
        static void position2dCamera (AbstractVehicle& selected);
        static void position2dCamera (AbstractVehicle& selected,
                                      float distance,
                                      float elevation);

        // camera updating utility used by several (all?) plug-ins
        static void updateCamera (const float currentTime,
                                  const float elapsedTime,
                                  const AbstractVehicle& selected);

        // some camera-related default constants
        static const float camera2dElevation;
        static const float cameraTargetDistance;
        static const Vec3 cameraTargetOffset;

        
        // @todo Clean this up and put it into the private section!
        static Graphics::OpenGlRenderService renderService_;
        static SharedPointer< Graphics::GraphicsAnnotationLogger > annotationLogger_;
        static SharedPointer< Graphics::Renderer > renderer_;
        // static SharedPointer< Graphics::BatchingRenderFeeder > renderFeeder_;
		// static SharedPointer< Graphics::LockingRenderFeeder > annotationRenderFeeder_;
		static SharedPointer< Graphics::ThreadStorageRenderFeeder::ThreadStorageType > threadStorage_;
		static SharedPointer< Graphics::ThreadStorageRenderFeeder > renderFeeder_;
        static bool annotationsOn_;
        
        
        static bool annotationsOn() {
            return annotationsOn_;
        }
        
        static void enableAnnotations() {
            annotationsOn_ = true;
            annotationLogger_->setRenderFeeder(     renderFeeder_.get() );
        }
        
        static void disableAnnotations() {
            annotationsOn_ = false;
            annotationLogger_->setRenderFeeder( 0 );
        }
        
        static void toggleAnnotationState() {
            if ( annotationsOn_ ) {
                disableAnnotations();
            } else {
                enableAnnotations();
            }
        }
        
		
		static void toggleClock() {
			OpenSteer::OpenSteerDemo::printMessage ( clock.togglePausedState () ?
                                                    "pause" : "run" );
			
			if ( measure_time_ ) {
				if ( clock.getPausedState() ) {
					demo_stop_watch_.suspend();
				} else {
					demo_stop_watch_.resume();
				}
				
			}
		}
		
		
		static void toggleMeasureTime() {
			
			std::ostringstream message;
			message << "Toggle measuring time ";
			if ( OpenSteer::OpenSteerDemo::measure_time_ ) {
				message << "off" << std::endl;
			} else {
				message << "on" << std::endl;
			}
			printMessage( message );
			
			if ( measure_time_ ) {
				demo_stop_watch_.suspend();
				kapaga::omp_stop_watch::time_type measured_time = demo_stop_watch_.elapsed_time();
				double const demo_time_s = kapaga::convert_time_to_s< double >( measured_time );
				
				measure_time_ = false;
				measure_time_until_max_frame_counter_ = false;
				
				double const aggregated_time_plugin_updates_s = kapaga::convert_time_to_s< double >( plugin_update_stop_watch_.elapsed_time() );
				double const aggregated_time_plugin_redraws_s = kapaga::convert_time_to_s< double >( plugin_redraw_stop_watch_.elapsed_time() );
				double const aggregated_time_demo_and_plugin_redraws_s = kapaga::convert_time_to_s< double >( demo_and_plugin_redraw_stop_watch_.elapsed_time() );
				
				double const aggregated_time_plugin_update_simulation_sub_stage = kapaga::convert_time_to_s< double >( selectedPlugIn->elapsed_time_simulation_stop_watch() );
				double const aggregated_time_plugin_update_modification_sub_stage = kapaga::convert_time_to_s< double >( selectedPlugIn->elapsed_time_modification_stop_watch() );
				double const aggregated_time_plugin_update_modification_sub_stage_agent_updates = kapaga::convert_time_to_s< double >( selectedPlugIn->elapsed_time_agent_update_stop_watch() );
				double const aggregated_time_plugin_update_modification_sub_stage_spatial_data_structures_updates = kapaga::convert_time_to_s< double >( selectedPlugIn->elapsed_time_spatial_data_structure_update_stop_watch() );
				
				std::clog << "Selected plugin: " << selectedPlugIn->name() << std::endl;
				#if defined(_OPENMP)
					std::clog << "OpenMP enabled" << std::endl;
					std::clog << "omp_get_num_procs:   " << omp_get_num_procs() << std::endl;
					std::clog << "omp_get_max_threads: " << omp_get_max_threads() << std::endl;
				#else
					std::clog << "OpenMP disabled" << std::endl;
				#endif
				std::clog << "Frames counted: " << plugin_frame_counter_ << std::endl;
				std::clog << "Aggreagated time for demo: " << demo_time_s << " [s]" << std::endl;
				std::clog << "Aggregated time for plugin update: " << aggregated_time_plugin_updates_s << " [s]" << std::endl;
				std::clog << "    Aggregated time for plugin update simulation sub-stage: " << aggregated_time_plugin_update_simulation_sub_stage << " [s]" << std::endl;
				std::clog << "    Aggregated time for plugin update modification sub-stage: " << aggregated_time_plugin_update_modification_sub_stage << " [s]" << std::endl;
				std::clog << "        Aggregated time for plugin update simulation sub-stage agent updates: " << aggregated_time_plugin_update_modification_sub_stage_agent_updates << " [s]" << std::endl;
				std::clog << "        Aggregated time for plugin update simulation sub-stage spatial data structure update: " << aggregated_time_plugin_update_modification_sub_stage_spatial_data_structures_updates << " [s]" << std::endl;
				std::clog << "Aggregated time for plugin redraw: " << aggregated_time_plugin_redraws_s << " [s]" << std::endl;
				std::clog << "Aggregated time for demo and plugin redraw: " << aggregated_time_demo_and_plugin_redraws_s << " [s]" << std::endl;
				
				
				
				std::clog << "Avg. time for demo: " << demo_time_s / plugin_frame_counter_ << " [s/frames]" << std::endl;
				std::clog << "Avg. time for plugin update: " << aggregated_time_plugin_updates_s / plugin_frame_counter_<< " [s/frames]" << std::endl;
				std::clog << "    Avg. time for plugin update simulation sub-stage: " << aggregated_time_plugin_update_simulation_sub_stage / plugin_frame_counter_ << " [s/frames]" << std::endl;
				std::clog << "    Avg. time for plugin update modification sub-stage: " << aggregated_time_plugin_update_modification_sub_stage / plugin_frame_counter_ << " [s/frames]" << std::endl;
				std::clog << "        Avg. time for plugin update simulation sub-stage agent updates: " << aggregated_time_plugin_update_modification_sub_stage_agent_updates / plugin_frame_counter_ << " [s/frames]" << std::endl;
				std::clog << "        Avg. time for plugin update simulation sub-stage spatial data structure update: " << aggregated_time_plugin_update_modification_sub_stage_spatial_data_structures_updates / plugin_frame_counter_ << " [s/frames]" << std::endl;
				std::clog << "Avg. time for plugin redraw: " << aggregated_time_plugin_redraws_s / plugin_frame_counter_ << " [s/frames]" << std::endl;
				std::clog << "Avg. time for demo and plugin redraw: " << aggregated_time_demo_and_plugin_redraws_s / plugin_frame_counter_ << " [s/frames ]" << std::endl;
				
				
				
				
				std::clog << "Avg. fps for demo: " << plugin_frame_counter_ / demo_time_s << " [fps]" << std::endl;
				std::clog << "Avg. fps plugin update: " <<  plugin_frame_counter_ / aggregated_time_plugin_updates_s << " [fps]" << std::endl;
				std::clog << "    Avg. fps for plugin update simulation sub-stage: " << plugin_frame_counter_ / aggregated_time_plugin_update_simulation_sub_stage << " [fps]" << std::endl;
				std::clog << "    Avg. fps for plugin update modification sub-stage: " << plugin_frame_counter_ / aggregated_time_plugin_update_modification_sub_stage << " [fps]" << std::endl;
				std::clog << "        Avg. fps for plugin update simulation sub-stage agent updates: " << plugin_frame_counter_ / aggregated_time_plugin_update_modification_sub_stage_agent_updates << " [fps]" << std::endl;
				std::clog << "        Avg. fps for plugin update simulation sub-stage spatial data structure update: " << plugin_frame_counter_ / aggregated_time_plugin_update_modification_sub_stage_spatial_data_structures_updates << " [fps]" << std::endl;
				std::clog << "Avg. fps plugin redraw: " <<  plugin_frame_counter_ / aggregated_time_plugin_redraws_s << " [fps]" << std::endl;
				std::clog << "Avg. fps demo and plugin redraw: " <<  plugin_frame_counter_ / aggregated_time_demo_and_plugin_redraws_s << " [fps]" << std::endl;
				
			} else {
				if ( ! clock.getPausedState() ) {
					demo_stop_watch_.restart( kapaga::omp_stop_watch::start_running );
				} else {
					demo_stop_watch_.restart( kapaga::omp_stop_watch::start_suspended );
				}
				measure_time_ = true;
				plugin_frame_counter_ = 0;
				plugin_update_stop_watch_.restart( kapaga::omp_stop_watch::start_suspended );
				plugin_redraw_stop_watch_.restart( kapaga::omp_stop_watch::start_suspended );
				demo_and_plugin_redraw_stop_watch_.restart( kapaga::omp_stop_watch::start_suspended );
				selectedPlugIn->reset_stop_watches();
			}
		}
        
		
		static void restartMeasuringTimeUntilSpecifiedFrameCount() 
		{
			std::ostringstream message;
			message << "Restart measuring time for a specified frame count. Restarts normal time measuring and stops it when the frame count is reached." << std::endl;
			printMessage( message );
						
			// Stop measuring time if active.
			if ( measure_time_ ) {
				toggleMeasureTime();
			}
			
			// Restart measuring time but with a max frame counter.
			measure_time_until_max_frame_counter_ = true;
			toggleMeasureTime();
		}
		
		
		
        static Graphics::OpenGlRenderFeeder::InstanceId floorGraphicsPrimitiveId;
        
        
        // ------------------------------------------------ graphics and annotation

        // do all initialization related to graphics
        static void initializeGraphics (void);

        // ground plane grid-drawing utility used by several plug-ins
        static void gridUtility (const Vec3& gridTarget);

        // draws a gray disk on the XZ plane under a given vehicle
        static void highlightVehicleUtility (const AbstractVehicle& vehicle);

        // draws a gray circle on the XZ plane under a given vehicle
        static void circleHighlightVehicleUtility (const AbstractVehicle& vehicle);

        // draw a box around a vehicle aligned with its local space
        // xxx not used as of 11-20-02
        static void drawBoxHighlightOnVehicle (const AbstractVehicle& v,
                                               const Color& color);

        // draws a colored circle (perpendicular to view axis) around the center
        // of a given vehicle.  The circle's radius is the vehicle's radius times
        // radiusMultiplier.
        static void drawCircleHighlightOnVehicle (const AbstractVehicle& v,
                                                  const float radiusMultiplier,
                                                  const Color& color);

        // ----------------------------------------------------------- console text

        // print a line on the console with "OpenSteerDemo: " then the given ending
        static void printMessage (const char* message);
        static void printMessage (const std::ostringstream& message);

        // like printMessage but prefix is "OpenSteerDemo: Warning: "
        static void printWarning (const char* message);
        static void printWarning (const std::ostringstream& message);

        // print list of known commands
        static void keyboardMiniHelp (void);

        // ---------------------------------------------------------------- private

    private:
        static int phase;
        static int phaseStack[];
        static int phaseStackIndex;
        static float phaseTimers[];
        static float phaseTimerBase;
        static const int phaseStackSize;
        static void pushPhase (const int newPhase);
        static void popPhase (void);
        static void initPhaseTimers (void);
        static void updatePhaseTimers (void);

        // XXX apparently MS VC6 cannot handle initialized static const members,
        // XXX so they have to be initialized not-inline.
        // static const int drawPhase = 2;
        // static const int updatePhase = 1;
        // static const int overheadPhase = 0;
        static const int drawPhase;
        static const int updatePhase;
        static const int overheadPhase;
		
		// Measure time for plugin updates and redrawing?
		static bool measure_time_;
		// Measure time for a given count of frames?
		static bool measure_time_until_max_frame_counter_;
		// If measuring time for a given count of frames - how many frames should be counted?
		static std::size_t const measure_time_frame_counter_max_;
		// Count the frames the plugin is updated or redrawn between open, close, reset calls.
		static std::size_t plugin_frame_counter_;
		// Stop watch to measure and aggregate the time needed to update a plugin.
		static kapaga::omp_stop_watch plugin_update_stop_watch_;
		// Stop watch to measure and aggregate the time needed to redraw a plugin.
		static kapaga::omp_stop_watch plugin_redraw_stop_watch_;
		// Stop watch to measure and aggregate the time needed to redraw the demo.
		static kapaga::omp_stop_watch demo_and_plugin_redraw_stop_watch_;
		// Measure the whole wall clock time the app is running while time should be measured.
		static kapaga::omp_stop_watch demo_stop_watch_;
        
    };

    // ----------------------------------------------------------------------------
    // do all initialization related to graphics

    /**
     * Initilaizes the graphics systems.
     *
     * @attention Call it before initializing @c OpenSteerDemo.
     */
    void initializeGraphics (int argc, char **argv);


    // ----------------------------------------------------------------------------
    // run graphics event loop


    void runGraphics (void);


    // ----------------------------------------------------------------------------
    // accessors for GLUT's window dimensions


    float drawGetWindowHeight (void);
    float drawGetWindowWidth (void);

} // namespace OpenSteer
    
    
// ----------------------------------------------------------------------------


// @todo Remove this ASAP.
#include "OpenSteer/Draw.h"


// ----------------------------------------------------------------------------
#endif // OPENSTEER_OPENSTEERDEMO_H
