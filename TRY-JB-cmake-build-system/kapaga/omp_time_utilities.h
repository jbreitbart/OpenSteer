/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

/**
 * @file
 *
 * Routines to convert time measured by @c omp_get_wtime into different time units and routines to
 * calculate the duration between to time points.
 */

//Header guard macro format: KAPAGA_namespace_file_name_H
#ifndef KAPAGA_kapaga_omp_time_utilities_H
#define KAPAGA_kapaga_omp_time_utilities_H

// Include KAPAGA_ASSERT
#include "kapaga/assert.h"

namespace kapaga {
	
	
	typedef double omp_time_t;
	
	// @todo Should these be public constants, public functions, or alltogether private?
	// Factor to convert seconds into milliseconds.
	extern omp_time_t const omp_time_factor_s_to_ms;
	// Factor to convert milliseconds to microseconds.
	extern omp_time_t const omp_time_factor_ms_to_us;
	// Factor to convert seconds into microsedonds.
	extern omp_time_t const omp_time_factor_s_to_us;

	// Factor to convert microseconds to milliseconds.
	extern omp_time_t const omp_time_factor_us_to_ms;
	// Factor to convert milliseconds to seconds.
	extern omp_time_t const omp_time_factor_ms_to_s;
	// Factor to convert microseconds to seconds.
	extern omp_time_t const omp_time_factor_us_to_s;
	
	
	
	/**
	 * Time duration between @a now and @a past.
	 *
	 * @pre @a now must be newer or equal to @a past.
	 */
	inline omp_time_t duration( omp_time_t now, omp_time_t past ) {
		KAPAGA_ASSERT( now >= past  && 
					   "Error: now must be newer or equal to past, otherwise behavior is undefined." );
		return now - past;
	}
	
	
	
	
	/**
	 * Converts the time measured by @c omp_get_wtime to milliseconds.
	 *
	 * @attention Doesn't take the resolution of @c omp_timer into account.
	 */
	template< typename ReturnType >
	ReturnType convert_time_to_ms( omp_time_t elapsed_time ) 
	{
		return static_cast< ReturnType >( omp_time_factor_s_to_ms * elapsed_time );
	}


	/**
	 * Converts the time returned by @c omp_get_wtime to seconds.
	 *
	 * @attention Doesn't take the resolution of @c omp_timer into account.
	 */
	template< typename ReturnType >
	ReturnType convert_time_to_s( omp_time_t elapsed_time ) 
	{
		return static_cast< ReturnType >( elapsed_time );
	}
	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_omp_time_utilities_H
