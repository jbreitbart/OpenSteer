#include "cupp/common.h"
#include "cupp/deviceT/vector.h"

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "ds/deviceT/gpu_grid.h"

using namespace OpenSteer::deviceT;


__global__ void preprocess(
                           const cupp::deviceT::vector<Vec3>  & positions,
                                 ds::deviceT::gpu_grid        & grid,
                                 unsigned int                   nb_of_boids
                          ) {
	__shared__ unsigned int cell_size[grid_nb_of_cells];

	{ // fill cell_size with 0's
		for (unsigned int i = threadIdx.x; i < grid_nb_of_cells; i += threads_per_block) {
			cell_size[i] = 0;
		}
		__syncthreads();
	}

	__shared__ Vec3 sh_positions[threads_per_block];
	unsigned int sh_base = 0;


	while (sh_base < nb_of_boids) {
		sh_positions[threadIdx.x] = positions[sh_base + threadIdx.x];
		__syncthreads();

		if (threadIdx.x==0) {
			for (unsigned int i = 0; i<threads_per_block; ++i) {
				++cell_size [grid.get_index( grid.get_cell_index(sh_positions[i]) )];
			}
		}
		
		sh_base += threads_per_block;
		
		__syncthreads();
	}

	if (threadIdx.x==0) {
		unsigned int value = 0;
		
		for (unsigned int i=1; i < grid_nb_of_cells; ++i) {
			value += cell_size[i-1];
			grid.index_[i] = value;
		}
	}
	
	if (threadIdx.x==42) {
		grid.index_[0] = 0;
		grid.index_[grid_nb_of_cells] = nb_of_boids;
	}

	__syncthreads();

	{ // fill cell_size with 0's
		for (unsigned int i = threadIdx.x; i < grid_nb_of_cells; i += threads_per_block) {
			cell_size[i] = 0;
		}
		__syncthreads();
	}

	if (threadIdx.x==0) {
		for (int i=0; i<nb_of_boids; ++i) {
			const int pos = grid.get_index( grid.get_cell_index(positions[i]) );
			grid.data_[grid.index_[pos]+cell_size[pos]] = i;
			++cell_size[pos];
		}
	}
}


preprocess_gridT get_preprocess_grid_kernel() {
	return (preprocess_gridT)preprocess;
}

