/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/os_process_utilities.h"


// Include usleep, sleep
#include <unistd.h>


// Include KAPAGA_ASSERT
#include "kapaga/assert.h"

// Include kapaga::clamp
#include "kapaga/math_utilities.h"


kapaga::os_process_sleep_us_t const kapaga::os_process_sleep_us_min = 0;
kapaga::os_process_sleep_us_t const kapaga::os_process_sleep_us_max = 1000000 - 1;


namespace {
	
	kapaga::os_process_sleep_ms_t const s_to_ms_factor = 1000;
	kapaga::os_process_sleep_ms_t const ms_to_us_factor = 1000;
	kapaga::os_process_sleep_ms_t const s_to_us_factor = s_to_ms_factor * ms_to_us_factor;
	
} // anonymous namespace



kapaga::os_process_sleep_s_t 
kapaga::os_process_sleep_s( os_process_sleep_s_t seconds )
{
	return sleep( seconds );
}


int kapaga::os_process_sleep_us_fast( os_process_sleep_us_t useconds )
{
	KAPAGA_ASSERT( useconds <= os_process_sleep_us_max && 
				   "Error: useconds must be less than 1,000,000, otherwise -1 is returned, an error number is set and the behavior is undefined." );
	return usleep( static_cast< useconds_t >( useconds ) );
}


//  @todo Does this need any error handling?
void kapaga::os_process_sleep_us( os_process_sleep_us_t useconds )
{
	os_process_sleep_us_t const usecs_clamped = clamp( useconds, os_process_sleep_us_min, os_process_sleep_us_max );
	os_process_sleep_us_fast( usecs_clamped );
}



kapaga::os_process_sleep_ms_t
kapaga::os_process_sleep_ms( kapaga::os_process_sleep_ms_t mseconds )
{
	/*
	os_process_sleep_ms_t const ms_in_us = mseconds * ms_to_us_factor;
	os_process_sleep_ms_t const count_os_process_sleep_us_max_in_mseconds = ms_in_us / os_process_sleep_us_max;
	os_process_sleep_ms_t const rest = ms_in_us % os_process_sleep_us_max;
	
	for ( os_process_sleep_ms_t i = 0; i < count_os_process_sleep_us_max_in_mseconds; ++i ) {
		os_process_sleep_us( os_process_sleep_us_max );
	}
	
	os_process_sleep_us( static_cast< os_process_sleep_us_t >( rest ) );
	 */
	
	// Calls sleep (in seconds) and usleep (in microseconds) to sleep for @a mseconds milliseconds.
	// @todo Add a compile time assertion to be sure that @c os_process_sleep_ms_t is an integer type.
	
	os_process_sleep_s_t const sleep_s = static_cast< os_process_sleep_s_t >( mseconds / s_to_ms_factor );

	os_process_sleep_ms_t const sleep_ms = mseconds % s_to_ms_factor; 
	
	os_process_sleep_us_t const sleep_us = static_cast< os_process_sleep_us_t >( ms_to_us_factor * sleep_ms );
	
	
	os_process_sleep_us_fast( static_cast< os_process_sleep_us_t >( sleep_us ) );
	return s_to_ms_factor * static_cast< os_process_sleep_ms_t >( os_process_sleep_s( sleep_s ) );
}


