#include "cupp/common.h"
#include "cupp/deviceT/vector.h"

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "ds/deviceT/gpu_grid.h"

using OpenSteer::deviceT::Vec3;

// algorithm based on the approach by mark harris et. al. see GPU Gems 3, pp. 851 - 876
__global__ void v1_prescan (
                                 ds::deviceT::gpu_grid        & grid,
                                 unsigned int                   n
                           ) {

	__shared__ int in[grid_nb_of_cells];
	for (int i=threadIdx.x; i<grid_nb_of_cells; i+=blockDim.x) {
		in[i] = grid.index_[i];
		grid.index_used_[i] = 0;
	}
	__syncthreads();
	
	if (threadIdx.x==0) {
		unsigned int value = 0;
		

		
		grid.index_[0] = 0;
		for (unsigned int i=1; i < grid_nb_of_cells; ++i) {
			value += in[i-1];
			grid.index_[i] = value;
		}
		grid.index_[grid_nb_of_cells] = n;
		
	}

#if 0
	__shared__ int temp[grid_nb_of_cells];

	temp[2*threadIdx.x]     = grid.index_[2*threadIdx.x];
	temp[2*threadIdx.x + 1] = grid.index_[2*threadIdx.x + 1];

	int offset = 1;

	for (int d = grid_nb_of_cells >> 1; d>0; d >>=1) {
		__syncthreads();
		if (threadIdx.x < d) {
			int ai = offset*(2*threadIdx.x+1)-1;
			int bi = offset*(2*threadIdx.x+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadIdx.x == 0) {
		temp[grid_nb_of_cells-1] = 0;
		grid.index_[grid_nb_of_cells] = n; 
	}

	for (int d=1; d<grid_nb_of_cells; d *= 2) {
		offset >>= 1;
		__syncthreads();

		if (threadIdx.x < d) {
			int ai = offset*(2*threadIdx.x+1)-1;
			int bi = offset*(2*threadIdx.x+2)-1;
			int t  = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	grid.index_[2*threadIdx.x] = temp[2*threadIdx.x];
	grid.index_[2*threadIdx.x + 1] = temp[2*threadIdx.x + 1];
#endif
}


v1_prescanT get_v1_prescan_kernel() {
	return (v1_prescanT)v1_prescan;
}
