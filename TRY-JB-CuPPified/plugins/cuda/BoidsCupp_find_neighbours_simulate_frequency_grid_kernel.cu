#include "cupp/deviceT/vector.h"
#include "cupp/common.h"
#include "OpenSteer/deviceT/Vec3.h"

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

#include "ds/deviceT/grid.h"
#include "OpenSteer/PosIndexPair.h"


using OpenSteer::PosIndexPair;
using namespace OpenSteer::deviceT;

__device__ int3 operator+ (const int3& rhs, const int3& lhs) {
	return make_int3 (rhs.x + lhs.x, rhs.y + lhs.y, rhs.z + lhs.z);
}


/*__device__ __constant__ int grid_data [7680];
int* get_grid_data_symbol() {
	return grid_data;
}*/

/* __device__ __constant__ int grid_index[grid_nb_of_cells+1];

int* get_grid_index_symbol() {
	return grid_index;
} */

#if 0
__global__ void find_neighbours_simulate_fr_gr (const int                              start,
                                                const ds::deviceT::grid<int>          &grid_,
                                                const cupp::deviceT::vector< Vec3  >  &positions_,
                                                const cupp::deviceT::vector< Vec3  >  &forwards_,
                                                      cupp::deviceT::vector< Vec3 >   &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;
	__shared__ ds::deviceT::grid<int> grid;
	grid = grid_;

	// constants that are needed below
	const unsigned int my_index        = start + blockIdx.x*blockDim.x + threadIdx.x;
	
	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];

	// get our position inside the grid
	const int3 my_grid_index   = grid.get_cell_index(position);

	__shared__ int   neighbours[neighbour_size_max * threads_per_block];
	__shared__ float neighbours_distance_squared[neighbour_size_max * threads_per_block];

	const int shared_base = threadIdx.x * neighbour_size_max;
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[shared_base+i]=-1;
	}

	// how many cell we will check in every direction
	const int cells_per_direction = rintf( r/grid.cell_size() );

	for (int x_offset = -cells_per_direction; x_offset<=cells_per_direction; ++x_offset) {
		for (int y_offset = -cells_per_direction; y_offset<=cells_per_direction; ++y_offset) {
			for (int z_offset = -cells_per_direction; z_offset<=cells_per_direction; ++z_offset) {

				const int3 cell_offset = make_int3(x_offset, y_offset, z_offset);
				const int3 cell        = my_grid_index + cell_offset;
				
				if (cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= grid.number_of_cells_per_dimension() || cell.y >= grid.number_of_cells_per_dimension() || cell.z >= grid.number_of_cells_per_dimension()) {
					continue;
				}

				const int end = grid_index[grid.get_index(cell)+1];
				for (int i = grid_index[grid.get_index(cell)]; i < end; ++i) {
					const int cur_index = grid.data_[i];
					const Vec3 offset = position - positions[cur_index];
					const float d2 = offset.lengthSquared();
					

					if (d2 < r2 && cur_index != my_index) {
						if (neighbours_found < neighbour_size_max) {
							neighbours[shared_base+neighbours_found]          = cur_index;
							neighbours_distance_squared[shared_base+neighbours_found] = d2;
							++neighbours_found;
						} else {
						
							float max_neighbour_distance = 0.0f;
							int max_neighbour_distance_index = 0;
							for ( int j = 0; j < neighbour_size_max; ++j ) {
								const float dist = neighbours_distance_squared[shared_base+j];
								if ( dist > max_neighbour_distance ) {
									max_neighbour_distance = dist;
									max_neighbour_distance_index = j;
								}
							}
							if (max_neighbour_distance>d2) {
								neighbours[shared_base+max_neighbour_distance_index]          = cur_index;
								neighbours_distance_squared[shared_base+max_neighbour_distance_index] = d2;
							}
						}
					}
				}
			}
		}
	}


	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	for (int i=0; i<neighbour_size_max; ++i) {

		if (neighbours[shared_base+i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}

		const float dist = neighbours_distance_squared[shared_base+i];
		
		if ( dist < boid_radius*3.0f) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = (position - positions[ neighbours[shared_base+i]]) / sqrtf (dist);
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

		int index = neighbours[shared_base+i];
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
#endif


#if 0
__global__ void find_neighbours_simulate_fr_gr (const int                              start,
                                                const ds::deviceT::grid<int>          &grid_,
                                                const cupp::deviceT::vector< Vec3  >  &positions_,
                                                const cupp::deviceT::vector< Vec3  >  &forwards_,
                                                      cupp::deviceT::vector< Vec3 >   &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;
	__shared__ ds::deviceT::grid<int> grid;
	grid = grid_;

	// constants that are needed below
	const unsigned int my_index        = start + blockIdx.x*blockDim.x + threadIdx.x;
	
	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];

	// get our position inside the grid
	const int3 my_grid_index   = grid.get_cell_index(position);

	__shared__ int   neighbours[neighbour_size_max * threads_per_block];
	__shared__ float neighbours_distance_squared[neighbour_size_max * threads_per_block];

	const int shared_base = threadIdx.x * neighbour_size_max;
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[shared_base+i]=-1;
	}

	// how many cell we will check in every direction
	const int cells_per_direction = rintf( r2/grid.cell_size() );

	__shared__ int grid_indexes [grid_nb_of_cells + 1];
	for (int i=threadIdx.x; i<grid_nb_of_cells+1; i+=threads_per_block) {
		grid_indexes[i] = grid.get_index(i);
	}
	__syncthreads();

	for (int x_offset = -cells_per_direction; x_offset<=cells_per_direction; ++x_offset) {
		for (int y_offset = -cells_per_direction; y_offset<=cells_per_direction; ++y_offset) {
			for (int z_offset = -cells_per_direction; z_offset<=cells_per_direction; ++z_offset) {

				const int3 cell_offset = make_int3(x_offset, y_offset, z_offset);
				const int3 cell        = my_grid_index + cell_offset;
				
				if (cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= grid.number_of_cells_per_dimension() || cell.y >= grid.number_of_cells_per_dimension() || cell.z >= grid.number_of_cells_per_dimension()) {
					continue;
				}

				const int end = grid_indexes[grid.get_index(cell)+1];
				for (int i = grid_indexes[grid.get_index(cell)]; i < end; ++i) {
					const int cur_index = grid.data_[i];
					const Vec3 offset = position - positions[cur_index];
					const float d2 = offset.lengthSquared();
					

					if (d2 < r2 && cur_index != my_index) {
						if (neighbours_found < neighbour_size_max) {
							neighbours[shared_base+neighbours_found]          = cur_index;
							neighbours_distance_squared[shared_base+neighbours_found] = d2;
							++neighbours_found;
						} else {
						
							float max_neighbour_distance = 0.0f;
							int max_neighbour_distance_index = 0;
							for ( int j = 0; j < neighbour_size_max; ++j ) {
								const float dist = neighbours_distance_squared[shared_base+j];
								if ( dist > max_neighbour_distance ) {
									max_neighbour_distance = dist;
									max_neighbour_distance_index = j;
								}
							}
							if (max_neighbour_distance>d2) {
								neighbours[shared_base+max_neighbour_distance_index]          = cur_index;
								neighbours_distance_squared[shared_base+max_neighbour_distance_index] = d2;
							}
						}
					}
				}
			}
		}
	}


	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	for (int i=0; i<neighbour_size_max; ++i) {

		if (neighbours[shared_base+i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}

		const float dist = neighbours_distance_squared[shared_base+i];
		
		if ( dist < boid_radius*3.0f) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = (position - positions[ neighbours[shared_base+i]]) / sqrtf (dist);
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

		int index = neighbours[shared_base+i];
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
#endif


__global__ void find_neighbours_simulate_fr_gr (const int                              start,
                                                const ds::deviceT::grid<int> &grid_,
                                                const cupp::deviceT::vector< Vec3  >  &positions_,
                                                const cupp::deviceT::vector< Vec3  >  &forwards_,
                                                      cupp::deviceT::vector< Vec3 >   &steering_results_)
{
	// make local copies of our references
	__shared__ cupp::deviceT::vector< Vec3 > positions;
	positions = positions_;
	__shared__ cupp::deviceT::vector< Vec3 > forwards;
	forwards = forwards_;
	__shared__ cupp::deviceT::vector< Vec3 > steering_results;
	steering_results = steering_results_;
	__shared__ ds::deviceT::grid<int> grid;
	grid = grid_;

	// constants that are needed below
	const unsigned int my_index        = start + blockIdx.x*blockDim.x + threadIdx.x;
	
	int neighbours_found = 0;
	
	const Vec3 position = positions[my_index];
	const Vec3 forward  = forwards[my_index];

	// get our position inside the grid
	const int3 my_grid_index   = grid.get_cell_index(position);

	__shared__ int   neighbours[neighbour_size_max * threads_per_block];
	__shared__ float neighbours_distance_squared[neighbour_size_max * threads_per_block];

	const int shared_base = threadIdx.x * neighbour_size_max;
	
	for (int i=0; i<neighbour_size_max; ++i) {
		neighbours[shared_base+i]=-1;
	}

	// how many cell we will check in every direction
	const int cells_per_direction = rintf( r2/grid.cell_size() );

	for (int x_offset = -cells_per_direction; x_offset<=cells_per_direction; ++x_offset) {
		for (int y_offset = -cells_per_direction; y_offset<=cells_per_direction; ++y_offset) {
			for (int z_offset = -cells_per_direction; z_offset<=cells_per_direction; ++z_offset) {

				const int3 cell_offset = make_int3(x_offset, y_offset, z_offset);
				const int3 cell        = my_grid_index + cell_offset;
				
				if (cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= grid.number_of_cells_per_dimension() || cell.y >= grid.number_of_cells_per_dimension() || cell.z >= grid.number_of_cells_per_dimension()) {
					continue;
				}
		
				for (const int* iter = grid.begin(cell); iter < grid.end(cell); ++iter) {
					const int cur_index = *iter;
					const Vec3 offset = position - positions[cur_index];
					const float d2 = offset.lengthSquared();
					

					if (d2 < r2 && cur_index != my_index) {
						if (neighbours_found < neighbour_size_max) {
							neighbours[shared_base+neighbours_found]          = cur_index;
							neighbours_distance_squared[shared_base+neighbours_found] = d2;
							++neighbours_found;
						} else {
						
							float max_neighbour_distance = 0.0f;
							int max_neighbour_distance_index = 0;
							for ( int j = 0; j < neighbour_size_max; ++j ) {
								const float dist = neighbours_distance_squared[shared_base+j];
								if ( dist > max_neighbour_distance ) {
									max_neighbour_distance = dist;
									max_neighbour_distance_index = j;
								}
							}
							if (max_neighbour_distance>d2) {
								neighbours[shared_base+max_neighbour_distance_index]          = cur_index;
								neighbours_distance_squared[shared_base+max_neighbour_distance_index] = d2;
							}
						}
					}
				}
			}
		}
	}


	bool do_seperation[neighbour_size_max];
	bool do_alignment[neighbour_size_max];
	bool do_cohesion[neighbour_size_max];

	for (int i=0; i<neighbour_size_max; ++i) {

		if (neighbours[shared_base+i]==-1) {
			do_seperation[i] = false;
			do_alignment[i]  = false;
			do_cohesion[i]   = false;
			continue;
		}

		const float dist = neighbours_distance_squared[shared_base+i];
		
		if ( dist < boid_radius*3.0f) {
			do_seperation[i] = true;
			do_alignment[i]  = true;
			do_cohesion[i]   = true;
			continue;
		}

		const Vec3  unitOffset  = (position - positions[ neighbours[shared_base+i]]) / sqrtf (dist);
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

		int index = neighbours[shared_base+i];
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



simulate_frequency_grid_kernelT get_find_neighbours_simulate_frequency_grid_kernel() {
	return (simulate_frequency_grid_kernelT)find_neighbours_simulate_fr_gr;
}
