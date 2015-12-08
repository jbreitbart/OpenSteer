/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/posix/omp_stubs.h"

/* CLOCKS_PER_SEC */
#include <time.h>

/* Include gettimeofday, timeval */
#include <sys/time.h>


/* Include KAPAGA_UNUSED_RETURN_VALUE */
#include "kapaga/unused_parameter.h"



/* @todo Add a test if gettimeofday really needs no error handling here. */
double omp_get_wtime( void )
{
	double secs = 0.0;
	struct timeval tv = { 0, 0};
	/* No error handling because the right types are handed into @c gettimeofday. */
	KAPAGA_UNUSED_RETURN_VALUE( gettimeofday( &tv, 0 ) );
	
	/* Convert seconds and microseconds to seconds. */
	secs = (double) tv.tv_sec;
	secs += (double)( tv.tv_usec ) / 1000000.0;
	
	return secs;
}


double omp_get_wtick( void )
{
	return 1.0 / (double) CLOCKS_PER_SEC;
}
