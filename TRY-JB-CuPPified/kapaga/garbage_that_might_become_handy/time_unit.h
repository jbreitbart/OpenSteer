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
 * Experimental code for expressing time units. Just for internal use. Don't use this file directly!
 */

#ifndef KAPAGA_kapaga_time_unit_H
#define KAPAGA_kapaga_time_unit_H

namespace kapaga {
	
	// @todo Rewrite using class tokens (like @c nothrow) to allow function overloading?
	// @todo Add cpu_clock_ticks??
	enum time_unit { years, weeks, days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds };
	
	/*
	struct time_unit_seconds { static time_unit unit() const; };
	struct time_unit_milliseconds { static time_unit unit() const; };
	struct time_unit_microseconds { static time_unit unit() const; };
	
	
	extern time_unit_seconds const time_in_seconds;
	extern time_unit_milliseconds const time_in_milliseconds;
	extern time_unit_microseconds const time_in_microseconds;
	*/
	
	
} // namespace kapaga

#endif // KAPAGA_kapaga_time_unit_H
