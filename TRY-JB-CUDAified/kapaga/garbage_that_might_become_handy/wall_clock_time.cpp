/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/posix/wall_clock_time.h"

// Include KAPAGA_UNUSED_RETURN_VALUE
#include "kapaga/unused_paramter.h"

// Include KAPAGA_ASSERT
#include "kapaga/assert.h"


kapaga::wall_clock_time_unit_t const kapaga::wall_clock_time_unit_factor_s_to_ms = static_cast< kapaga::wall_clock_time_unit_t >( 1000 );
kapaga::wall_clock_time_unit_t const kapaga::wall_clock_time_unit_factor_ms_to_us = static_cast< kapaga::wall_clock_time_unit_t >( 1000 );	
kapaga::wall_clock_time_unit_t const kapaga::wall_clock_time_unit_factor_s_to_us = kapaga::wall_clock_time_unit_factor_s_to_ms * kapaga::wall_clock_time_unit_factor_ms_to_us;

kapaga::wall_clock_time_unit_t const kapaga::wall_clock_time_unit_factor_us_to_ms = static_cast< kapaga::wall_clock_time_unit_t >( 1 ) / kapaga::wall_clock_time_unit_factor_ms_to_us;
kapaga::wall_clock_time_unit_t const kapaga::wall_clock_time_unit_factor_ms_to_s = static_cast< kapaga::wall_clock_time_unit_t >( 1 ) / kapaga::wall_clock_time_unit_factor_s_to_ms;	
kapaga::wall_clock_time_unit_t const kapaga::wall_clock_time_unit_factor_us_to_s = static_cast< kapaga::wall_clock_time_unit_t >( 1 ) / kapaga::wall_clock_time_unit_factor_s_to_us;




kapaga::wall_clock_time_t 
kapaga::wall_clock_time_now()
{
	timeval tv = { 0, 0};
	KAPAGA_UNUSED_RETURN_VALUE( gettimeofday( &tv, 0 ) );
	
	return tv;
}


kapaga::wall_clock_time_t 
kapaga::duration( wall_clock_time_t const& now, 
				  wall_clock_time_t const& past )
{
	KAPAGA_ASSERT( now >= past && 
				   "Error: now must be later or equal to past, otherwise behavior is undefined.");
	
	timeval tv( now );
	tv.tv_sec - past.tv_sec;
	tv.tv_usec - past.tv_usec;
	
	return tv;
}


bool operator==( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs )
{
	
	NORMALIZE??
	
	return ( lhs.tv_sec == rhs.tv_sec ) && ( lhs.tv_usec == rhs.tv_usec );
}


bool operator!=( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs )
{
	return ! operator==( lhs, rhs );
}


bool operator<( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs )
{
	NORMALIZE????????
	
	bool lesser = true;
	
	if ( ( lhs.tv_sec > rhs.tv_sec ) ||  
		 ( ( lhs.tv_sec == rhs.tv_sec ) && 
			   ( lhs.tv_usec >= rhs.tv_usec ) ) {
		lesser = false;
	}
	
	return lesser;
}


bool operator<=( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs )
{
	return ! operator<( rhs, lhs );
}


bool operator>( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs )
{
	return operator<( rhs, lhs );
}


bool operator>=( wall_clock_time_t const& lhs, wall_clock_time_t const& rhs )
{
	return ! operator<( lhs, rhs );
}

