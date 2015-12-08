/**
 * @file
 *
 * Stubs for some OpenMP functions used in OpenSteer if no OpenMP capable compiler is used.
 * See the OpenMP Application Program Interface, Version 2.5 May 2005, pp. 203--208.
 */

#ifndef OPENSTEER_OMPSTUBS_H
#define OPENSTEER_OMPSTUBS_H

#ifdef __cplusplus
extern "C" {
#endif
	
	/**
	 * @return @c 1 as there is one (and only one) thread in a sequential app.
	 */
	inline int omp_get_max_threads()
	{
		return 1;
	}

	
	/**
	 * @return @c 0. @c 0 is the standard id of the one and only thread in a sequential app.
	 */
	inline int omp_get_thread_num()
	{
		return 0;
	}	
	
	
#ifdef __cplusplus
// Close extern "C" scope.
}
#endif


#endif // OPENSTEER_OMPSTUBS_H
