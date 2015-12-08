/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/omp_stop_watch.h"


// Include omp_get_wtime
#include "kapaga/omp_header_wrapper.h"


// Includes kapaga::duration
#include "kapaga/omp_time_utilities.h"


namespace {
	
	typedef kapaga::omp_stop_watch::time_type time_type;
	
	inline time_type elapsed_time_helper( time_type current_time,
										  time_type time_stamp,
										  time_type elapsed_time,
										  bool running ) {
		time_type currently_elapsed_time = elapsed_time;
		
		if ( running ) {
			currently_elapsed_time += kapaga::duration( current_time, time_stamp );
		}
		
		return currently_elapsed_time;
	}
	
	
} // anonymous namespace


// Initialize static member variables of @c omp_stop_watch.
kapaga::omp_stop_watch::start_suspended_type const kapaga::omp_stop_watch::start_suspended = kapaga::omp_stop_watch::start_suspended_type();
kapaga::omp_stop_watch::start_running_type const kapaga::omp_stop_watch::start_running = kapaga::omp_stop_watch::start_running_type();




kapaga::omp_stop_watch::omp_stop_watch( start_running_type const& )
: time_stamp_( omp_get_wtime() ), elapsed_time_( time_type( 0 ) ), running_( true )
{
	// The actual parameter isn't important, just that the overloaded constructor that starts
	// in running mode is called is important.
	
	// Nothing to do.
}


kapaga::omp_stop_watch::omp_stop_watch( start_suspended_type const& )
: time_stamp_( omp_get_wtime() ), elapsed_time_( time_type( 0 ) ), running_( false )
{
	// The actual parameter isn't important, just that the overloaded constructor that starts
	// in suspended mode is called is important.
	
	// Nothing to do.
}


void 
kapaga::omp_stop_watch::restart( start_running_type const& )
{
	// The actual parameter isn't important, just that the overloaded member function that restarts
	// in running mode is called is important.
	
	time_stamp_ = omp_get_wtime();
	elapsed_time_ = time_type( 0 );
	running_ = true;
	
}



void 
kapaga::omp_stop_watch::restart( start_suspended_type const& )
{
	// The actual parameter isn't important, just that the overloaded member function that restarts
	// in suspended mode is called is important.
	
	time_stamp_ = omp_get_wtime();
	elapsed_time_ = time_type( 0 );
	running_ = false;
	
}



void 
kapaga::omp_stop_watch::suspend()
{
	time_type current_time = omp_get_wtime();
	elapsed_time_ = elapsed_time_helper( current_time, time_stamp_, elapsed_time_, running_ );
	// time_stamp_ = current_time; // @todo Is it necessary to store the current time stamp?
	running_ = false;
}


void 
kapaga::omp_stop_watch::resume()
{
	if ( ! running_ ) {
		time_stamp_ = omp_get_wtime();
		running_ = true;
	}
}




kapaga::omp_stop_watch::time_type 
kapaga::omp_stop_watch::elapsed_time() const
{	
	return elapsed_time_helper( omp_get_wtime(), time_stamp_, elapsed_time_, running_ );
}



