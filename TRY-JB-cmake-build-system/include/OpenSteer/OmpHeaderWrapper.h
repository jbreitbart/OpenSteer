#ifndef OPENSTEER_OMPHEADERWRAPPER_H
#define OPENSTEER_OMPHEADERWRAPPER_H

#ifdef _OPENMP
#include <omp.h>
#else
#include "OpenSteer/OmpStubs.h"
#endif


#endif // OPENSTEER_OMPHEADERWRAPPER_H
