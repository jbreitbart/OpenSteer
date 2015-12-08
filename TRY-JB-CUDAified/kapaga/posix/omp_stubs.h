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
 * Stubs for some OpenMP functions if no OpenMP capable compiler is used.
 *
 * See the OpenMP Application Program Interface, Version 2.5 May 2005, pp. 203--208.
 *
 * @todo Should the function declarations be separated from their definitions?
 */

#ifndef KAPAGA_omp_stubs_H
#define KAPAGA_omp_stubs_H

#ifdef __cplusplus
extern "C" {
#endif
	
	
	/**
	 * @return @c 1 as there is one (and only one) thread in a sequential app.
	 */
	int omp_get_max_threads( void );
	inline int omp_get_max_threads( void )
	{
		return 1;
	}

	/**
	 * @return @c 1 as there is one (and only one) thread in a sequential app that can currently be
	 *         active.
	 */
	int omp_get_num_threads( void );
	inline int omp_get_num_threads( void )
	{
		return 1;
	}

	/**
	 * @return @c 1 as there is one processor available for OpenMP if nt compiled with an OpenMP
	 *         supporting compiler.
	 */
	int omp_get_num_procs( void );
	inline int omp_get_num_procs( void )
	{
		return 1;
	}



	/**
	 * @return @c 0. @c 0 is the standard id of the one and only thread in a sequential app.
	 */
	int omp_get_thread_num( void );
	inline int omp_get_thread_num( void )
	{
		return 0;
	}	


	/**
	 * @return @c 1 or @c true if called from within a parallel region, @c 0 or @c false otherwise.
	 */
	int omp_in_parallel( void );
	inline int omp_in_parallel( void )
	{
		return 0;
	}


	/**
	 * Measures the elapsed wall clock time from a moment in the past.
	 * To measure the time between two moments call @c omp_get_wtime twice and subtract the values
	 * from each other.
	 *
	 * @return elapsed wall clock time in seconds measured from a moment in the past. If the
	 *         environment can not provide the time behavior is undefined.
	 */
	double omp_get_wtime( void );

	/**
	 * Returns the precision of the timer used by @c omp_get_wtime.
	 *
	 * @return a value equal to the number of seconds between successive clock ticks of the timer
	 *         used by @c omp_get_wtime.
	 */
	double omp_get_wtick( void );

#ifdef __cplusplus
/* Close extern "C" scope. */
}
#endif


#endif /* KAPAGA_omp_stubs_H */
