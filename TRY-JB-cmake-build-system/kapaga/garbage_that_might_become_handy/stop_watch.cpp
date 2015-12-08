/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "stop_watch.h"


// Include KAPAGA_ASSERT
#include "kapaga/assert.h"

// Include omp_get_wtime
#include "kapga/omp_header_wrapper.h"


namespace {
	
	using kapaga::timer::time_type;
	
	// Factor to multiply the duration with to get ms instead of seconds.
	time_type const time_to_ms_factor = 1000.0;
	
	
	
	inline time_type convert_time_to_ms( time_type time ) {
		return time * time_to_ms_factor;
	}
	
	inline time_typee current_wall_clock_time() {
		return omp_get_wtime();
	}
	
	
	/**
	 * Duration from @a past to @a now.
	 */
	inline time_type duration( time_type now, time_type past ) {
		KAPAGA_ASSERT( now >= past  && 
					   "Error: now must be greater or equal to past, otherwise behavior is undefined." );
		return now - past;
	}
	
	
	inline time_type elapsed_time_helper( time_type current_time,
										  time_type time_stamp,
										  time_type elapsed_time,
										  bool running ) {
		time_type currently_elapsed_time = elapsed_time;
		
		if ( running_ ) {
			currently_elapsed_time += duration( current_time, time_stamp );
		}
		
		return currently_elapsed_time;
	}
	

	
	
} // anonymous namespace


	

	
kapaga::stop_watch::stop_watch()
: time_stamp_( current_wall_clock_time() ), elapsed_time_( time_type( 0 ) ), running_( true )
{
	// Nothing to do.
}


	
void 
kapaga::stop_watch::restart()
{
	time_stamp_ = current_wall_clock_time();
	elapsed_time_ = time_type( 0 );
	running_ = true;
	
}



void 
kapaga::stop_watch::suspend()
{
	time_type current_time = current_wall_clock_time();
	elapsed_time_ = elapsed_time_helper( current_time, time_stamp_, elapsed_time_, running_ );
	time_stamp_ = current_time;
	running_ = false;
}


void 
kapaga::stop_watch::resume()
{
	if ( ! running_ ) {
		time_stamp_ = current_wall_clock_time();
		running_ = true;
	}
}




kapaga::stop_watch::time_type 
kapaga::stop_watch::elapsed_time() const
{	
	return convert_time_to_ms( elapsed_time_helper( current_wall_clock_time(), time_stamp_, elapsed_time_, running_ ) );
}
	

