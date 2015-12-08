#ifndef OPENSTEER_cuppconfig_H
#define OPENSTEER_cuppconfig_H

#include "cupp/common.h"

// CUDA
#include <vector_types.h>

#include "OpenSteer/CompileTimeAssertion.h"
#include <stddef.h>

#if !defined(NVCC)
#include "cupp/device.h"
#endif

// I know this looks strange, but the cuda compiler needs these values
// at compile time ... so this solution or macros
namespace {

const unsigned int neighbour_size_max = 7;
const unsigned int threads_per_block  = 128;

/***  MAGIC VARIABLES FOR BOIDS CLASS  ***/
const float boid_worldRadius      = 200.0f;
const float boid_mass             = 1.0f;
const float boid_radius           = 0.5f;
const float boid_maxForce         = 27.0f;
const float boid_maxSpeed         = 9.0f;
const float boid_maxAdjustedSpeed = 0.2f * boid_maxSpeed;

const float separationRadius = 5.0f;
const float separationAngle  = -0.707f;
const float separationWeight = 12.0f;

const float alignmentRadius  = 7.5f;
const float alignmentAngle   = 0.7f;
const float alignmentWeight  = 8.0f;

const float cohesionRadius   = 9.0f;
const float cohesionAngle    = -0.15f;
const float cohesionWeight   = 8.0f;

const float r                = 9.0f;
const float r2               = 18.0f;

// right now grid_nb_of_cells has to be a multiply of threads_per_block!
const int unsigned grid_size = 30;
const int unsigned grid_nb_of_cells = grid_size * grid_size * grid_size;

}

namespace OpenSteer {


// Yeah ... more global variables
// well if _someone_ wants to extend the OpenSteer infrastructure to
// manually pass them to plugins do it but this is fine right now
#if !defined(NVCC)
extern const cupp::device cupp_device;
extern const unsigned int no_of_multiprocessors; // I am wondering, why I can't query this from the device...
extern const unsigned int think_frequency;

/**
 * We should at have at least one block for every multiprocessor ... better much more
 */
extern const size_t gBoidStartCount;

/**
 * How many vehicles to add when pressing the right key?
 * Use a multiply of @a threads_per_block (this make code much easier ;-))
 */
extern const size_t add_vehicle_count;

#endif

}

#endif
