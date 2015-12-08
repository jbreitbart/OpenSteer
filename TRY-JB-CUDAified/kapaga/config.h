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
 * Definitions to detect the target platform.
 * 
 * Standard defines are documented here: http://predef.sourceforge.net
 *
 * The idea of a config header can be found in projects using autotools and also in UnitTest++
 * (where I got the idea from, see http://unittest-cpp.sourceforge.net ).
 *
 * @todo Rewrite it to be really usable. Take a look or use poshlib.
 */

#ifndef KAPAGA_config_H
#define KAPAGA_config_H

#if defined( __APPLE__ ) && defined( __MACH__ )
#define KAPAGA_MAC_OSX
#endif

#if defined( linux ) || defined( __linux )
#define KAPAGA_LINUX
#endif

#if defined(KAPAGA_MAC_OSX ) || defined(KAPAGA_LINUX)
#define KAPAGA_POSIX
#endif


#endif // KAPAGA_config_H
