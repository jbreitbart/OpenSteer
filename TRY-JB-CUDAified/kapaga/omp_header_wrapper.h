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
 * Header wrapper that includes the OpenMP header if compiled by an OpenMP capable compiler
 * (with OpenMP enabled) or includes a few OpenMP function stubs if compiled with a compiler
 * that does not know OpenMP.
 */

#ifndef KAPAGA_omp_header_wrapper_H
#define KAPAGA_omp_header_wrapper_H

#ifdef _OPENMP
	#include <omp.h>
#else
	#include "kapaga/omp_stubs.h"
#endif


#endif // KAPAGA_omp_header_wrapper_H
