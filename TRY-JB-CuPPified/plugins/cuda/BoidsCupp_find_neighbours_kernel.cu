#include "cupp/deviceT/vector.h"
#include "cupp/common.h"
#include "OpenSteer/deviceT/Vec3.h"
#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

using OpenSteer::deviceT::Vec3;

#if 0
// called V3
__global__ void find_neighbours (const cupp::deviceT::vector< Vec3  > &positions_,
                                 const float                           r2,
                                       cupp::deviceT::vector< int > &find_neighbours_result_)
{
	// make local copies of our references
	const cupp::deviceT::vector< Vec3  > positions(positions_);
	      cupp::deviceT::vector< int > find_neighbours_result(find_neighbours_result_);

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	
	int result[neighbour_size_max];
	for (int i=0; i<neighbour_size_max; ++i) {
		result[i]=-1;
	}
	
	__shared__ Vec3 s_positions[threads_per_block];
	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			float const d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index) {
				if (neighbours_found < neighbour_size_max) {
					result[neighbours_found] = cur_index;
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						float const dist = ( position - positions[ result[j] ] ).lengthSquared();

						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						result[max_neighbour_distance_index] = cur_index;
					}
				}
			}
			++i;
		}

		__syncthreads();
	}
	
	const int result_index_base = my_index*neighbour_size_max;
	for (int i=0; i<neighbour_size_max; ++i) {
		find_neighbours_result[result_index_base + i] = result[i];
	}
}
#endif

#if 0
// called V2
// the one with the strange behavior (fast on second run)
__global__ void find_neighbours (const cupp::deviceT::vector< Vec3  > &positions_,
                                 const float                           r2,
                                       cupp::deviceT::vector< int > &find_neighbours_result_)
{
	// make local copies of our references
	const cupp::deviceT::vector< Vec3  > positions(positions_);
	      cupp::deviceT::vector< int > find_neighbours_result(find_neighbours_result_);

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;

	// use shared here reduces the number of registers need, which results
	// in the possibility of more blocks per multiprocessor and a higher occupancy of each multiprocessor
	__shared__ int number_of_boids;
	 number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];

	int result[neighbour_size_max];
	float result_distance[neighbour_size_max];
	for (int i=0; i<neighbour_size_max; ++i) {
		result[i]=-1;
	}

	int max_neighbour_distance_index = 0;
	float max_neighbour_distance = 0.0f;
	
	__shared__ Vec3 s_positions[threads_per_block];

	for (int base=0; base<number_of_boids; base+=threads_per_block) {

		// read positions from global to shared memory
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		for (int i=0; i<threads_per_block; ++i) {
			const float d2 = (position - s_positions[i]).lengthSquared();

			// fill variables with dummy values we can write in every cycle
			int   cur_index = result[0];
			int   result_index = 0;
			float cur_neighbour_distance = result_distance[0];

			if (d2 < r2 && (base + i != my_index) ) {
				if (neighbours_found < neighbour_size_max) {
					cur_neighbour_distance = d2;
					cur_index = base + i;
					result_index = neighbours_found;
					if (max_neighbour_distance<d2) {
						max_neighbour_distance = d2;
					}
					++neighbours_found;
				} else {
					if (d2 < max_neighbour_distance) {
						cur_neighbour_distance = d2;
						cur_index = base + i;
						result_index = max_neighbour_distance_index;
					}
				}
			}

			// write result
			result [result_index]          = cur_index;
			result_distance[result_index]  = cur_neighbour_distance;

			// update max_neighbour_distance & index
			if (d2 < max_neighbour_distance) {
				for (int j=0; j<neighbour_size_max; ++j) {
					if (result_distance[j] > max_neighbour_distance) {
						max_neighbour_distance = result_distance[j];
						max_neighbour_distance_index = j;
					}
				}
			}
		}

		__syncthreads();
	}

	// write the result to global memory
	const int result_index_base = my_index*neighbour_size_max;
	for (int i=0; i<neighbour_size_max; ++i) {
		find_neighbours_result[result_index_base + i] = result[i];
	}
}
#endif


#if 0
// Paper: first GPU version
__global__ void find_neighbours (const cupp::deviceT::vector< Vec3  > &positions_,
                                 const float                           r2,
                                       cupp::deviceT::vector< int > &find_neighbours_result_)
{
	const cupp::deviceT::vector< Vec3 > positions               = positions_;
	      cupp::deviceT::vector< int > find_neighbours_result  = find_neighbours_result_;

	const unsigned int my_index = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	
	int neighbours_found = 0;
	
	float max_neighbour_distance = 0.0f;
	int max_neighbour_distance_index = 0;

	const Vec3 position = positions[my_index];
	
	int result[neighbour_size_max];
	for (int i=0; i<neighbour_size_max; ++i) {
		result[i]=-1;
	}

	int i=0;
	while (i < number_of_boids && neighbours_found<neighbour_size_max) {
		const Vec3 offset = position - positions[i];
		float const d2 = offset.lengthSquared();
		if (d2<r2 && i!=my_index) {
			if ( d2 > max_neighbour_distance ) {
				max_neighbour_distance = d2;
				max_neighbour_distance_index = neighbours_found;
			}
			result[neighbours_found] = i;
			++neighbours_found;
		}
		++i;
	}
	
	while (i < number_of_boids) {
		const Vec3 offset = position - positions[i];
		float const d2 = offset.lengthSquared();
		if (d2<r2 && d2 < max_neighbour_distance && i != my_index) {
			result[ max_neighbour_distance_index ] = i;
			max_neighbour_distance = d2; // just temporary

			for ( int i = 0; i < neighbour_size_max; ++i ) {
				float const dist = ( position - positions[ result[i] ] ).lengthSquared();

				if ( dist > max_neighbour_distance ) {
					max_neighbour_distance = dist;
					max_neighbour_distance_index = i;
				}
			}
		}
		++i;
	}
	
	for (int i=0; i<neighbour_size_max; ++i) {
		find_neighbours_result[my_index*neighbour_size_max + i] = result[i];
	}
}
#endif



// paper version with shared memory
__global__ void find_neighbours (const cupp::deviceT::vector< Vec3  > &positions_,
                                 const float                                               r2,
                                       cupp::deviceT::vector< int >                       &find_neighbours_result_)
{
	// make local copies of our references
	const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  > positions(positions_);
	      cupp::deviceT::vector< int > find_neighbours_result(find_neighbours_result_);

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	// use shared here reduces the number of registers need, which results
	// in the possibility of more blocks per multiprocessor and a higher occupancy of each multiprocessor
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	
	int result[neighbour_size_max];
	float result_distance[neighbour_size_max];
	for (int i=0; i<neighbour_size_max; ++i) {
		result[i]=-1;
	}
	
	__shared__ Vec3 s_positions[threads_per_block];
	for (int base=0; base < number_of_boids; base+=threads_per_block) {
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
					result[neighbours_found] = cur_index;
					result_distance[neighbours_found] = d2;
					++neighbours_found;
				} else {
					float max_neighbour_distance = 0.0f;
					int max_neighbour_distance_index = 0;
					for ( int j = 0; j < neighbour_size_max; ++j ) {
						const float dist = result_distance[j];
						if ( dist > max_neighbour_distance ) {
							max_neighbour_distance = dist;
							max_neighbour_distance_index = j;
						}
					}
					if (max_neighbour_distance>d2) {
						result[max_neighbour_distance_index] = cur_index;
						result_distance[max_neighbour_distance_index] = d2;
					}
				}
			}
			++i;
		}

		__syncthreads();
	}
	
	const int result_index_base = my_index*neighbour_size_max;
	for (int i=0; i<neighbour_size_max; ++i) {
		find_neighbours_result[result_index_base + i] = result[i];
	}
}


#if 0
// called V5
// less registeres, but slower performance compared to shared shard memory paper version
__global__ void find_neighbours (const cupp::deviceT::vector< Vec3  > &positions_,
                                 const float                           r2,
                                       cupp::deviceT::vector< int > &find_neighbours_result_)
{
	// make local copies of our references
	const cupp::deviceT::vector< Vec3  > positions(positions_);
	      cupp::deviceT::vector< int > find_neighbours_result(find_neighbours_result_);

	// constants that are needed below
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ unsigned int number_of_boids;
	number_of_boids = gridDim.x*blockDim.x; // number of boids == number of threads

	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	
	int result[neighbour_size_max];
	float result_distance[neighbour_size_max];
	for (int i=0; i<neighbour_size_max; ++i) {
		result[i]=-1;
	}
		
	float max_neighbour_distance = 0.0f;
	int max_neighbour_distance_index = 0;
	
	__shared__ Vec3 s_positions[threads_per_block];
	for (int base=0; base < number_of_boids; base+=threads_per_block) {
		s_positions[threadIdx.x] = positions[base + threadIdx.x];
		__syncthreads();

		int i=0;
		while (i < threads_per_block) {
			const Vec3 offset = position - s_positions[i];
			float const d2 = offset.lengthSquared();
			const int cur_index = base + i;
			
			if (d2 < r2 && cur_index != my_index && neighbours_found < neighbour_size_max) {
				result[neighbours_found] = cur_index;
				result_distance[neighbours_found] = d2;
				if (d2 > max_neighbour_distance) {
					max_neighbour_distance = d2;
					max_neighbour_distance_index = neighbours_found;
				}
				++neighbours_found;
			}
			if (d2 < r2 && cur_index != my_index && max_neighbour_distance>d2 && neighbours_found >= neighbour_size_max) {
				result[max_neighbour_distance_index] = cur_index;
				result_distance[max_neighbour_distance_index] = d2;
				for ( int j = 0; j < neighbour_size_max; ++j ) {
					const float dist = result_distance[j];
					if ( dist > max_neighbour_distance ) {
						max_neighbour_distance = dist;
						max_neighbour_distance_index = j;
					}
				}
			}
			++i;
		}

		__syncthreads();
	}
	
	const int result_index_base = my_index*neighbour_size_max;
	for (int i=0; i<neighbour_size_max; ++i) {
		find_neighbours_result[result_index_base + i] = result[i];
	}
}
#endif

find_neighbours_kernelT get_find_neighbours_kernel() {
	return (find_neighbours_kernelT)find_neighbours;
}
