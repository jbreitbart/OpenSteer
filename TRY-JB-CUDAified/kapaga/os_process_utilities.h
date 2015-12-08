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
 * Utility functions to help manage operating system processes from userland.
 */

// No header guards because this header just chooses the platform specific header that has
// header guards.

#include "kapaga/config.h"

#if defined(KAPAGA_POSIX)
	#include "kapaga/posix/os_process_utilities.h"
#else
	#error Unknown platform.
#endif

