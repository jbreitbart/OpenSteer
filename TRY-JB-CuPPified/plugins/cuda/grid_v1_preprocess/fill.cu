#include "cupp/common.h"
#include "cupp/deviceT/vector.h"

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "ds/deviceT/gpu_grid.h"

using OpenSteer::deviceT::Vec3;

__global__ void v1_fill (
                           const cupp::deviceT::vector<Vec3>  & positions,
                                 ds::deviceT::gpu_grid        & grid,
                                 unsigned int                   nb_of_boids
                        ) {
#if 0
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (my_index >= nb_of_boids) return;
	
	// cell index finden
	const Vec3 pos = positions[my_index];
	const int cell = grid.get_index(grid.get_cell_index(pos));
	
	// atomic + 1
	const int insert = atomicAdd(grid.index_used_+cell, 1);
	
	grid.data_[grid.index_[cell] + insert] = my_index;
#endif


	const int my_index  = blockIdx.x*threads_per_block + threadIdx.x;
	const int my_cell_x = my_index % grid_size;
	const int my_cell_y = (my_index / grid_size) % grid_size;
	const int my_cell_z = my_index / grid_size / grid_size;

	const float low_x  = my_cell_x * grid.cell_size_;
	const float high_x = (my_cell_x+1) * grid.cell_size_;
	
	const float low_y  = my_cell_y * grid.cell_size_;
	const float high_y = (my_cell_y+1) * grid.cell_size_;
	
	const float low_z  = my_cell_z * grid.cell_size_;
	const float high_z = (my_cell_z+1) * grid.cell_size_;
	
	int count = 0;
	int sh_base = 0;
	
	__shared__ Vec3 sh_positions[threads_per_block];

	const float world_size = grid.world_size_;

	const int global_base = grid.index_[my_index];

	while (sh_base < nb_of_boids) {
		sh_positions[threadIdx.x] = positions[sh_base + threadIdx.x] + world_size;
		__syncthreads();
		
		for (int i=0; i<threads_per_block; ++i) {
			Vec3 &p = sh_positions[i];
			if (p.x() >= low_x && p.x() < high_x &&
			    p.y() >= low_y && p.y() < high_y &&
			    p.z() >= low_z && p.z() < high_z) {
				grid.data_[global_base+count] = i+sh_base;
				++count;
			}
		}

		sh_base += threads_per_block;

		__syncthreads();
	}
}

v1_fillT get_v1_fill_kernel() {
	return (v1_fillT)v1_fill;
}
