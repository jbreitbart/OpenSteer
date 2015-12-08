/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_gpu_grid_H
#define DS_gpu_grid_H

#if defined(NVCC)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

// get our device type
#include "ds/deviceT/gpu_grid.h"

// get the stl vector
#include <vector>
#include <iostream>

#include "cupp/kernel_type_binding.h"
#include "cupp/device_reference.h"
#include "cupp/device.h"
#include "cupp/shared_device_pointer.h"

namespace ds {

/**
 * @class gpu_grid
 * @author Jens Breitbart
 * @version 0.1
 * @date 08.08.2007
 * @platform Host only
 * @brief More a handle to the grid data on the gpu, than a real grid
 */

class gpu_grid {
	public: /*** TYPEDEFS  ***/
		typedef ds::deviceT::gpu_grid                  device_type;
		typedef gpu_grid                               host_type;

	private:
		/**
		 * Dimension of a cell in each dimension
		 */
		const float cell_size_;

		/**
		 * The number of cells in each dimension; total number of cells == number_of_cells_per_dimension_³
		 */
		const int number_of_cells_per_dimension_;

		cupp::shared_device_pointer<int> *device_data_ptr_;
		const cupp::shared_device_pointer<int> device_index_;
		const cupp::shared_device_pointer<int> device_index_used_;

		const float world_size_;
	
	public:

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		explicit gpu_grid( const float world_size, const std::size_t number_of_cells_per_dimension )
		             :
		              cell_size_ ( world_size*2 / number_of_cells_per_dimension ),
		              number_of_cells_per_dimension_ ( number_of_cells_per_dimension ),
		              device_data_ptr_(0),
		              device_index_( cupp::malloc<int>(number_of_cells_per_dimension * number_of_cells_per_dimension * number_of_cells_per_dimension + 1)),
		              device_index_used_ ( cupp::malloc<int>(number_of_cells_per_dimension * number_of_cells_per_dimension * number_of_cells_per_dimension)),
		              world_size_(world_size)
		{}

		void resize(std::size_t size) {
			delete device_data_ptr_;
			int* temp = cupp::malloc<int>(size);
			device_data_ptr_ = new cupp::shared_device_pointer<int>( temp );
		}

	public: /*** CUPP FUNCTIONALITY  ***/

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		device_type transform (const cupp::device &/*d*/) {

			return device_type::construct(device_data_ptr_->get(), device_index_.get(), device_index_used_.get(), cell_size_, number_of_cells_per_dimension_, world_size_);
		}

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		cupp::device_reference< device_type > get_device_reference(const cupp::device &d) {
			return cupp::device_reference<device_type> (d, transform(d));
		}

		/**
		 * @brief This function is called by the kernel_call_traits
		 */
		void dirty (cupp::device_reference< device_type > /*device_copy*/) {
			// nothing to do ... we don't have data on the CPU, so why should we care of the device data is changed?
		}
};

}


#endif
