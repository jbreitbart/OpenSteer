#include "cupp/deviceT/vector.h"
#include "cupp/common.h"
#include "OpenSteer/deviceT/Vec3.h"

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

#include "ds/deviceT/dyn_grid_clever_v2.h"
#include "OpenSteer/PosIndexPair.h"


using OpenSteer::PosIndexPair;
using namespace OpenSteer::deviceT;
using ds::dyn_grid_clever_v2_node;

__device__ int3 operator+ (const int3& rhs, const int3& lhs) {
	return make_int3 (rhs.x + lhs.x, rhs.y + lhs.y, rhs.z + lhs.z);
}

__device__ void neighbour_search(
                                 const ds::deviceT::dyn_grid_clever_v2           &grid,
                                 const dyn_grid_clever_v2_node& neighbour_block, const dyn_grid_clever_v2_node& cur_block,
                                 const unsigned int& my_index, const Vec3 &position,
                                 int &neighbours_found, int *neighbours, float *neighbours_distance_squared
                                ) {

		
		__shared__ PosIndexPair s_PosIndex[threads_per_block];

		const int end = neighbour_block.size;
		for (int base=0; base < end; base+=threads_per_block) {
			if (base + threadIdx.x < end) {
				s_PosIndex[threadIdx.x] = grid.get_PosIndex(neighbour_block, base+threadIdx.x);
			}
			__syncthreads();

			if (threadIdx.x < cur_block.size) {

				int i=0;
				while ( base+i < end && i < threads_per_block) {
					const Vec3 offset = position - s_PosIndex[i].position();
					const float d2 = offset.lengthSquared();
					const int cur_index = s_PosIndex[i].index();

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
			}

			__syncthreads();
		}
}


__global__ void find_neighbours_simulate_dyn_gr_clever_v2 (
                                                 const ds::deviceT::dyn_grid_clever_v2 &grid_,
                                                 const cupp::deviceT::vector< Vec3  >  &positions,
                                                 const cupp::deviceT::vector< Vec3  >  &forwards,
                                                       cupp::deviceT::vector< Vec3 >   &steering_results)
{
	__shared__ ds::deviceT::dyn_grid_clever_v2 grid;
	grid = grid_;
	__shared__ dyn_grid_clever_v2_node cur_block;
	__shared__ dyn_grid_clever_v2_node neighbour_block;
	if (threadIdx.x == 0) {
		cur_block = grid.get_block_data(blockIdx.x);
		
	}
	if (threadIdx.x == 1) {
		neighbour_block = grid.get_area(blockIdx.x);
	}

	__syncthreads();

	// constants that are needed below

	unsigned int my_index;
	
	Vec3 position;
	Vec3 forward;
	int neighbours_found = 0;
	
	int   neighbours[neighbour_size_max];
	float neighbours_distance_squared[neighbour_size_max];
	
	if (threadIdx.x < cur_block.size) {
		my_index = grid.get_index(cur_block, threadIdx.x);
		
		position = grid.get_position(cur_block, threadIdx.x);
		forward   = forwards [my_index];
		
		for (int i=0; i<neighbour_size_max; ++i) {
			neighbours[i]=-1;
		}
	}

	__syncthreads();

	
	neighbour_search(grid, neighbour_block, cur_block, my_index, position, neighbours_found, neighbours, neighbours_distance_squared);

	if (threadIdx.x >= cur_block.size) {
		return;
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


find_neighbours_simulate_dyn_gr_clever_v2_kernelT get_find_neighbours_simulate_dyn_grid_clever_v2_kernel() {
	return (find_neighbours_simulate_dyn_gr_clever_v2_kernelT)find_neighbours_simulate_dyn_gr_clever_v2;
}
