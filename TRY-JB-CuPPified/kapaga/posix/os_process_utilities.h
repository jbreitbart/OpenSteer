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
 * @file os_process_utilities.h
 *
 * Posix utility functions to help manage operating system processes from userland.
 *
 * @todo Reread the posix standard for the @c 1,000,000 limit of @c usleep and platforms that don't 
 *       have it.
 */

//Header guard macro format: KAPAGA_namespace_file_name_H
#ifndef KAPAGA_kapaga_os_process_utilities_H
#define KAPAGA_kapaga_os_process_utilities_H

namespace kapaga {
	
	// @todo Replace with platform specific greatest value holding type.
	// Type to specify milliseconds for the @c os_process_sleep functions.
	typedef unsigned long os_process_sleep_ms_t;
	
	// Type to specify seconds for the @c os_process_sleep functions.
	typedef unsigned int os_process_sleep_s_t;
	
	// Type to specify microseconds for the @c os_process_sleep functions.
	typedef unsigned int os_process_sleep_us_t;
	
	/**
	 * Min value to pass to @c os_process_sleep_us or @c os_process_sleep_us_fast.
	 */
	extern os_process_sleep_us_t const os_process_sleep_us_min;
	
	/**
	 * Max value to pass to @c os_process_sleep_us or @c os_process_sleep_us_fast.
	 */
	extern os_process_sleep_us_t const os_process_sleep_us_max;
	
	
	/**
	 * Sleep for @a seconds seconds.
	 *
	 * @return the unslept amount of time if a signal occured while sleeping or @c 0 if the 
	 * requested time has elapsed.
	 *
	 * @attention Not of very high precision or reliability.
	 */
	os_process_sleep_s_t os_process_sleep_s( os_process_sleep_s_t seconds );
	
	/**
	 * Puts the process to sleep for @a useconds microseconds.
	 *
	 * @pre @a useconds must be lesser or equal to os_process_sleep_us_max.
	 *
	 * @return @c 0 if completed sleep successfully, otherwise @c -1 and sets an error number
	 *
	 * @todo Rework error handling to be more of a C++ style.
	 */
	int os_process_sleep_us_fast( os_process_sleep_us_t useconds );
	
	
	/**
	 * Puts the process to sleep for @a useconds microseconds.
	 *
	 * @pre @a useconds must be lesser or equal to os_process_sleep_us_max, otherwise it is clamped.
	 */
	void os_process_sleep_us( os_process_sleep_us_t useconds ); 
	
	
	/**
	 * Puts the process to sleep for @a mseconds milliseconds.
	 *
	 * @return number of milliseconds it hasn't slept, for example if a signal disturbed the sleep.
	 *
	 * @attention Not of very high precision or reliability.
	 */
	os_process_sleep_ms_t os_process_sleep_ms( os_process_sleep_ms_t mseconds );
	
} // namespace kapaga


#endif // KAPAGA_POSIX_kapaga_os_process_utilities_H 
