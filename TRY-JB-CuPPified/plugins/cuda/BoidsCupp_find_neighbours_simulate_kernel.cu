#include "cupp/deviceT/vector.h"
#include "cupp/common.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

using OpenSteer::deviceT::Vec3;


#if 0
// paper - store version
__global__ void find_neighbours_simulate (const cupp::deviceT::vector< Vec3  > &positions_,
                                          const cupp::deviceT::vector< Vec3  > &forwards_,
                                                cupp::deviceT::vector< Vec3 >  &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];
	
	int   neighbours[neighbour_size_max];
	float neighbours_distance_squared[neighbour_size_max];
	Vec3  neighbours_offset[neighbour_size_max];
	Vec3  neighbours_forward[neighbour_size_max];
	Vec3  neighbours_position[neighbour_size_max];
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[i]=-1;
	}

	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		__shared__ Vec3 s_positions[threads_per_block];
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			const float d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index) {
				if (neighbours_found < neighbour_size_max) {
					neighbours[neighbours_found]          = cur_index;
					neighbours_distance_squared[neighbours_found] = d2;
					neighbours_offset[neighbours_found]   = offset;
					neighbours_position[neighbours_found] = s_positions[i];
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						const float dist = neighbours_distance_squared[j];
						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						neighbours[max_neighbour_distance_index]          = cur_index;
						neighbours_distance_squared[max_neighbour_distance_index] = d2;
						neighbours_offset[max_neighbour_distance_index]   = offset;
						neighbours_position[max_neighbour_distance_index] = s_positions[i];
					}
				}
			}
			++i;
		}

		__syncthreads();
	}

	/// @todo try loop till neighbours_found and remove if <- requires more registers
	for (int i=0; i<neighbour_size_max; ++i) {
		if (neighbours[i]==-1) {
			break;
		}
		neighbours_forward[i] = forwards[ neighbours[i] ];
	}

	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	for (int i=0; i<neighbour_size_max; ++i) {

		if (neighbours[i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}
		
		/// @todo remove this if and put it into the calculation below
		if (neighbours_distance_squared[i] < boid_radius*3.0f) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = neighbours_offset[i] / sqrtf (neighbours_distance_squared[i]);
		const float forwardness = forward.dot (unitOffset);
		
		do_seperation[i] = (forwardness > separationAngle && neighbours_distance_squared[i] <= separationRadius*separationRadius);
		do_alignment[i]  = (forwardness > alignmentAngle  && neighbours_distance_squared[i] <= alignmentRadius*alignmentRadius);
		do_cohesion[i]   = (forwardness > cohesionAngle   && neighbours_distance_squared[i] <= cohesionRadius*cohesionRadius);
	}

	Vec3 separation = { 0.0f, 0.0f, 0.0f };
	Vec3 alignment  = { 0.0f, 0.0f, 0.0f };
	Vec3 cohesion   = { 0.0f, 0.0f, 0.0f };
	
	int influencing_alignment_neighbour_count = 0;
	int influencing_cohesion_neighbour_count  = 0;
	for (int i=0; i<neighbour_size_max; ++i) {
		
		/// @todo avoid obstacles if needed


		if (do_seperation[i]) {
			if ( 0.0f != neighbours_distance_squared[i] ) {
				separation = separation + (neighbours_offset[i] / neighbours_distance_squared[i]);
			} else {
				separation = separation + neighbours_offset[i];
			}
		}

		if (do_alignment[i]) {
			// accumulate sum of neighbor's heading
			alignment = alignment + neighbours_forward[i];

			// count neighbors
			++influencing_alignment_neighbour_count;
		}

		if (do_cohesion[i]) {
			// accumulate sum of neighbor's positions
			cohesion = cohesion + neighbours_position[i];

			// count neighbors
			++influencing_cohesion_neighbour_count;
		}
	}
	
	alignment = alignment - ( forward  * influencing_alignment_neighbour_count );
	cohesion  = cohesion  - ( position * influencing_cohesion_neighbour_count  );

	// apply weights to components (save in variables for annotation)
	const Vec3 separationW = separation.normalize() * separationWeight;
	const Vec3 alignmentW  = alignment.normalize()  * alignmentWeight;
	const Vec3 cohesionW   = cohesion.normalize()   * cohesionWeight;

	steering_results[my_index] = separationW + alignmentW + cohesionW;
}
#endif

#if 0
__global__ void find_neighbours_simulate (const cupp::deviceT::vector< Vec3  > &positions_,
                                          const cupp::deviceT::vector< Vec3  > &forwards_,
                                                cupp::deviceT::vector< Vec3 >  &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];
	
	int   neighbours[neighbour_size_max];
	float neighbours_distance_squared[neighbour_size_max];
	Vec3  neighbours_position[neighbour_size_max];
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[i]=-1;
	}

	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		__shared__ Vec3 s_positions[threads_per_block];
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			const float d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index) {
				if (neighbours_found < neighbour_size_max) {
					neighbours[neighbours_found]          = cur_index;
					neighbours_distance_squared[neighbours_found] = d2;
					neighbours_position[neighbours_found] = s_positions[i];
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						const float dist = neighbours_distance_squared[j];
						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						neighbours[max_neighbour_distance_index]          = cur_index;
						neighbours_distance_squared[max_neighbour_distance_index] = d2;
						neighbours_position[max_neighbour_distance_index] = s_positions[i];
					}
				}
			}
			++i;
		}

		__syncthreads();
	}

	

	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	//const int shared_base = threadIdx.x*neighbour_size_max;
	
	for (int i=0; i<neighbour_size_max; ++i) {

		/// @todo try loop till neighbours_found and remove if <- requires more registers
		if (neighbours[i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}
		
		if (neighbours_distance_squared[i] < boid_radius*3.0f) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = (position - neighbours_position[i]) / sqrtf (neighbours_distance_squared[i]);
		const float forwardness = forward.dot (unitOffset);
		
		do_seperation[i] = (forwardness > separationAngle && neighbours_distance_squared[i] <= separationRadius*separationRadius);
		do_alignment[i]  = (forwardness > alignmentAngle  && neighbours_distance_squared[i] <= alignmentRadius*alignmentRadius);
		do_cohesion[i]   = (forwardness > cohesionAngle   && neighbours_distance_squared[i] <= cohesionRadius*cohesionRadius);
	}

	Vec3 separation = { 0.0f, 0.0f, 0.0f };
	Vec3 alignment  = { 0.0f, 0.0f, 0.0f };
	Vec3 cohesion   = { 0.0f, 0.0f, 0.0f };
	
	int influencing_alignment_neighbour_count = 0;
	int influencing_cohesion_neighbour_count  = 0;
	for (int i=0; i<neighbour_size_max; ++i) {
		
		/// @todo avoid obstacles if needed


		if (do_seperation[i]) {
			if ( 0.0f != neighbours_distance_squared[i] ) {
				separation = separation + ((position - neighbours_position[i]) / neighbours_distance_squared[i]);
			} else {
				separation = separation + position - neighbours_position[i];
			}
		}

		if (do_alignment[i]) {
			// accumulate sum of neighbor's heading
			alignment = alignment + forwards[ neighbours[i] ];

			// count neighbors
			++influencing_alignment_neighbour_count;
		}

		if (do_cohesion[i]) {
			// accumulate sum of neighbor's positions
			cohesion = cohesion + neighbours_position[i];

			// count neighbors
			++influencing_cohesion_neighbour_count;
		}
	}
	
	alignment = alignment - ( forward  * influencing_alignment_neighbour_count );
	cohesion  = cohesion  - ( position * influencing_cohesion_neighbour_count  );

	// apply weights to components (save in variables for annotation)
	const Vec3 separationW = separation.normalize() * separationWeight;
	const Vec3 alignmentW  = alignment.normalize()  * alignmentWeight;
	const Vec3 cohesionW   = cohesion.normalize()   * cohesionWeight;

	steering_results[my_index] = separationW + alignmentW + cohesionW;
}
#endif

#if 0
__global__ void find_neighbours_simulate (const cupp::deviceT::vector< Vec3  > &positions_,
                                          const cupp::deviceT::vector< Vec3  > &forwards_,
                                                cupp::deviceT::vector< Vec3 >  &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];
	
	int   neighbours[neighbour_size_max];
	float neighbours_distance_squared[neighbour_size_max];
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[i]=-1;
	}

	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		__shared__ Vec3 s_positions[threads_per_block];
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			const float d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index) {
				if (neighbours_found < neighbour_size_max) {
					neighbours[neighbours_found]          = cur_index;
					neighbours_distance_squared[neighbours_found] = d2;
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						const float dist = neighbours_distance_squared[j];
						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						neighbours[max_neighbour_distance_index]          = cur_index;
						neighbours_distance_squared[max_neighbour_distance_index] = d2;
					}
				}
			}
			++i;
		}

		__syncthreads();
	}

	

	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	for (int i=0; i<neighbour_size_max; ++i) {

		/// @todo try loop till neighbours_found and remove if <- requires more registers
		if (neighbours[i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}

		const float dist = neighbours_distance_squared[i];
		
		if ( dist < boid_radius*3.0f) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = (position - positions[ neighbours[i]]) / sqrtf (dist);
		const float forwardness = forward.dot (unitOffset);
		
		do_seperation[i] = (forwardness > separationAngle && dist <= separationRadius*separationRadius);
		do_alignment[i]  = (forwardness > alignmentAngle  && dist <= alignmentRadius*alignmentRadius);
		do_cohesion[i]   = (forwardness > cohesionAngle   && dist <= cohesionRadius*cohesionRadius);
	}

	Vec3 separation = { 0.0f, 0.0f, 0.0f };
	Vec3 alignment  = { 0.0f, 0.0f, 0.0f };
	Vec3 cohesion   = { 0.0f, 0.0f, 0.0f };
	
	int influencing_alignment_neighbour_count = 0;
	int influencing_cohesion_neighbour_count  = 0;
	for (int i=0; i<neighbour_size_max; ++i) {
		
		/// @todo avoid obstacles if needed


		if (do_seperation[i]) {
			Vec3 temp = positions[neighbours[i]];
			if ( 0.0f != neighbours_distance_squared[i] ) {
				separation = separation + ((position - temp) / (position - temp).lengthSquared());
			} else {
				separation = separation + position - temp;
			}
		}

		if (do_alignment[i]) {
			// accumulate sum of neighbor's heading
			alignment = alignment + forwards[ neighbours[i] ];

			// count neighbors
			++influencing_alignment_neighbour_count;
		}

		if (do_cohesion[i]) {
			// accumulate sum of neighbor's positions
			cohesion = cohesion + positions[neighbours[i]];

			// count neighbors
			++influencing_cohesion_neighbour_count;
		}
	}
	
	alignment = alignment - ( forward  * influencing_alignment_neighbour_count );
	cohesion  = cohesion  - ( position * influencing_cohesion_neighbour_count  );

	// apply weights to components (save in variables for annotation)
	const Vec3 separationW = separation.normalize() * separationWeight;
	const Vec3 alignmentW  = alignment.normalize()  * alignmentWeight;
	const Vec3 cohesionW   = cohesion.normalize()   * cohesionWeight;

	steering_results[my_index] = separationW + alignmentW + cohesionW;
}
#endif

// much more registers and not faster
#if 0
__global__ void find_neighbours_simulate (const cupp::deviceT::vector< Vec3  > &positions_,
                                          const cupp::deviceT::vector< Vec3  > &forwards_,
                                                cupp::deviceT::vector< Vec3 >  &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];
	
	int   neighbours[neighbour_size_max];
	float neighbours_distance_squared[neighbour_size_max];
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[i]=-1;
	}

	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		__shared__ Vec3 s_positions[threads_per_block];
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			const float d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index) {
				if (neighbours_found < neighbour_size_max) {
					neighbours[neighbours_found]          = cur_index;
					neighbours_distance_squared[neighbours_found] = d2;
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						const float dist = neighbours_distance_squared[j];
						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						neighbours[max_neighbour_distance_index]          = cur_index;
						neighbours_distance_squared[max_neighbour_distance_index] = d2;
					}
				}
			}
			++i;
		}

		__syncthreads();
	}

	

	Vec3 separation = { 0.0f, 0.0f, 0.0f };
	Vec3 alignment  = { 0.0f, 0.0f, 0.0f };
	Vec3 cohesion   = { 0.0f, 0.0f, 0.0f };
	
	int influencing_alignment_neighbour_count = 0;
	int influencing_cohesion_neighbour_count  = 0;

	for (int i=0; i<neighbours_found; ++i) {
		
		const float dist = neighbours_distance_squared[i];
		const int index = neighbours[i];
		
		const Vec3  unitOffset  = (position - positions[index]) / sqrtf (dist);
		const float forwardness = forward.dot (unitOffset);
		
		bool do_seperation = dist < boid_radius*3.0f || (forwardness > separationAngle && dist <= separationRadius*separationRadius);
		bool do_alignment  = dist < boid_radius*3.0f || (forwardness > alignmentAngle  && dist <= alignmentRadius*alignmentRadius);
		bool do_cohesion   = dist < boid_radius*3.0f || (forwardness > cohesionAngle   && dist <= cohesionRadius*cohesionRadius);
	
		if (do_seperation) {
			Vec3 temp = positions[index];
			if ( 0.0f != dist ) {
				separation = separation + ((position - temp) / dist);
			} else {
				separation = separation + position - temp;
			}
		}

		if (do_alignment) {
			// accumulate sum of neighbor's heading
			alignment = alignment + forwards[ index ];

			// count neighbors
			++influencing_alignment_neighbour_count;
		}

		if (do_cohesion) {
			// accumulate sum of neighbor's positions
			cohesion = cohesion + positions[index];

			// count neighbors
			++influencing_cohesion_neighbour_count;
		}
	}

	alignment = alignment - ( forward  * influencing_alignment_neighbour_count );
	cohesion  = cohesion  - ( position * influencing_cohesion_neighbour_count  );

	// apply weights to components (save in variables for annotation)
	const Vec3 separationW = separation.normalize() * separationWeight;
	const Vec3 alignmentW  = alignment.normalize()  * alignmentWeight;
	const Vec3 cohesionW   = cohesion.normalize()   * cohesionWeight;

	steering_results[my_index] = separationW + alignmentW + cohesionW;
}
#endif



// Paper version - recalculate
__global__ void find_neighbours_simulate (const cupp::deviceT::vector< Vec3  > &positions_,
                                          const cupp::deviceT::vector< Vec3  > &forwards_,
                                                cupp::deviceT::vector< Vec3 >  &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];
	
	int   neighbours[neighbour_size_max];
	float neighbours_distance_squared[neighbour_size_max];
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[i]=-1;
	}

	
	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		__shared__ Vec3 s_positions[threads_per_block];
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		#pragma unroll 64
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			const float d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index) {
				if (neighbours_found < neighbour_size_max) {
					neighbours[neighbours_found]          = cur_index;
					neighbours_distance_squared[neighbours_found] = d2;
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						const float dist = neighbours_distance_squared[j];
						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						neighbours[max_neighbour_distance_index]          = cur_index;
						neighbours_distance_squared[max_neighbour_distance_index] = d2;
					}
				}
			}
			++i;
		}

		__syncthreads();
	}

	

	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	for (int i=0; i<neighbour_size_max; ++i) {

		/// @todo try loop till neighbours_found and remove if <- requires more registers
		if (neighbours[i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}

		const float dist = neighbours_distance_squared[i];
		
		if ( dist < boid_radius*3.0f ) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = (position - positions[ neighbours[i]]) / sqrtf (dist);
		const float forwardness = forward.dot (unitOffset);
		
		do_seperation[i] = (forwardness > separationAngle && dist <= separationRadius*separationRadius);
		do_alignment[i]  = (forwardness > alignmentAngle  && dist <= alignmentRadius*alignmentRadius);
		do_cohesion[i]   = (forwardness > cohesionAngle   && dist <= cohesionRadius*cohesionRadius);
	}

	Vec3 separation = { 0.0f, 0.0f, 0.0f };
	Vec3 alignment  = { 0.0f, 0.0f, 0.0f };
	Vec3 cohesion   = { 0.0f, 0.0f, 0.0f };
	
	int influencing_alignment_neighbour_count = 0;
	int influencing_cohesion_neighbour_count  = 0;
	for (int i=0; i<neighbour_size_max; ++i) {
		
		/// @todo avoid obstacles if needed

		int index = neighbours[i];
		if (do_seperation[i]) {
			Vec3 temp = position - positions[index];
			if ( 0.0f != temp.lengthSquared() ) {
				separation = separation + (temp / temp.lengthSquared());
			} else {
				separation = separation + temp;
			}
		}

		if (do_alignment[i]) {
			// accumulate sum of neighbor's heading
			alignment = alignment + forwards[ index ];

			// count neighbors
			++influencing_alignment_neighbour_count;
		}

		if (do_cohesion[i]) {
			// accumulate sum of neighbor's positions
			cohesion = cohesion + positions[index];

			// count neighbors
			++influencing_cohesion_neighbour_count;
		}
	}
	
	alignment = alignment - ( forward  * influencing_alignment_neighbour_count );
	cohesion  = cohesion  - ( position * influencing_cohesion_neighbour_count  );

	// apply weights to components (save in variables for annotation)
	const Vec3 separationW = separation.normalize() * separationWeight;
	const Vec3 alignmentW  = alignment.normalize()  * alignmentWeight;
	const Vec3 cohesionW   = cohesion.normalize()   * cohesionWeight;

	steering_results[my_index] = separationW + alignmentW + cohesionW;
}


simulate_kernelT get_find_neighbours_simulate_kernel() {
	return (simulate_kernelT)find_neighbours_simulate;
}
