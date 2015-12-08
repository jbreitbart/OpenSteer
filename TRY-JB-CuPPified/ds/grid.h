/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_grid_H
#define DS_grid_H

#if defined(NVCC)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

// get our device type
#include "ds/deviceT/grid.h"

// get the stl vector
#include <vector>
#include <iostream>

#include "cupp/kernel_type_binding.h"
#include "cupp/device_reference.h"
#include "cupp/device.h"

namespace ds {

/**
 * @class grid
 * @author Jens Breitbart
 * @version 0.1
 * @date 08.08.2007
 * @platform Host only
 * @brief A very simple 3-D grid implementation.
 * Requirements for template parameter T:
 * - T.position().{x|y|z} needs to give the {x|y|z}-coord of the position of the object
 */

template <typename T>
class grid {
	private:
		typedef typename cupp::get_type<T>::device_type             T_device_type;
		
	public: /*** TYPEDEFS  ***/
		typedef ds::deviceT::grid< T_device_type >                  device_type;
		typedef grid<T>                                             host_type;

	private:
		/**
		 * Well ... our grid :-)
		 */
		std::vector < std::vector < T > > data_;

		/**
		 * Dimension of a cell in each dimension
		 */
		const float cell_size_;

		/**
		 * The number of cells in each dimension; total number of cells == number_of_cells_per_dimension_³
		 */
		const int number_of_cells_per_dimension_;

		mutable cupp::memory1d<T_device_type> *device_data_ptr_;
		mutable cupp::memory1d<int>   *device_index_ptr_;

		const float world_size_;

		mutable std::vector<T_device_type> device_data_;
		mutable std::vector<int>   device_index_;

		const int *const_symbol_memory;
	public:

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		explicit grid( const float world_size, const std::size_t number_of_cells_per_dimension/*, const int* symbol_mem*/ )
		             :
		              data_ ( number_of_cells_per_dimension*number_of_cells_per_dimension*number_of_cells_per_dimension ),
		              cell_size_ ( world_size*2 / number_of_cells_per_dimension ),
		              number_of_cells_per_dimension_ ( number_of_cells_per_dimension ),
		              device_data_ptr_(0),
		              device_index_ptr_(0),
		              world_size_(world_size),
		              const_symbol_memory(0)//symbol_mem)
		{}

		/**
		 * This will insert an element into the grid
		 * @param inserted The element that will be inserted
		 */
		void insert ( const T& inserted, const OpenSteer::Vec3 &position ) {
			int where_to_insert  = static_cast <int> ( (position.x+world_size_) / cell_size_);
			where_to_insert     += static_cast <int> ( (position.y+world_size_) / cell_size_) * number_of_cells_per_dimension_;
			where_to_insert     += static_cast <int> ( (position.z+world_size_) / cell_size_) * number_of_cells_per_dimension_ * number_of_cells_per_dimension_;

			data_[where_to_insert].push_back(inserted);
		}

		/**
		 * Deletes all values from the grid
		 */
		void clear () {
			for (std::size_t i = 0; i<data_.size(); ++i) {
				data_[i].clear();
			}
			device_data_.clear();
			device_index_.clear();
		}
		
	public: /*** CUPP FUNCTIONALITY  ***/

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		device_type transform (const cupp::device &d) {
			int index_counter  = 0;

			for (std::size_t i=0; i < data_.size(); ++i) {
				device_index_.push_back(index_counter);
				
				const std::vector<T> &cur_cell = data_[i];
				copy (cur_cell.begin(), cur_cell.end(), back_inserter(device_data_));
				
				index_counter+=cur_cell.size();
			}
			device_index_.push_back(index_counter);

			if (device_data_ptr_ == 0 || device_data_ptr_ -> size() != data_.size()) {
				delete device_data_ptr_;
				device_data_ptr_ = new cupp::memory1d<T_device_type>(d, &device_data_[0], device_data_.size());
			} else {
				device_data_ptr_ -> copy_to_device (device_data_.size(), &device_data_[0]);
			}

			if (device_data_ptr_ == 0 || device_data_ptr_ -> size() != data_.size()) {
				delete device_index_ptr_;
				device_index_ptr_ = new cupp::memory1d<int>(d, &device_index_[0], device_index_.size());
			} else {
				device_index_ptr_ -> copy_to_device (device_index_.size(), &device_index_[0]);
			}

			/*if (cudaMemcpyToSymbol((const char*)const_symbol_memory, &device_index_[0], sizeof(int)*device_index_.size()) != cudaSuccess) {
				throw "doo";
			}*/
			
			return device_type::construct(device_data_ptr_->transform(d), device_index_ptr_->transform(d), cell_size_, number_of_cells_per_dimension_, world_size_);
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
			throw "Should never be called";
		}
};

}


#endif

#if 0
/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_grid_H
#define DS_grid_H

#if defined(NVCC)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

// get our device type
#include "ds/deviceT/grid.h"

// get the stl vector
#include <vector>
#include <iostream>

#include "cupp/kernel_type_binding.h"
#include "cupp/device_reference.h"
#include "cupp/device.h"

namespace ds {

/**
 * @class grid
 * @author Jens Breitbart
 * @version 0.1
 * @date 08.08.2007
 * @platform Host only
 * @brief A very simple 3-D grid implementation.
 * Requirements for template parameter T:
 * - T.position().{x|y|z} needs to give the {x|y|z}-coord of the position of the object
 */

template <typename T>
class grid {
	private:
		typedef typename cupp::get_type<T>::device_type             T_device_type;
		
	public: /*** TYPEDEFS  ***/
		typedef ds::deviceT::grid< T_device_type >                  device_type;
		typedef grid<T>                                             host_type;

	private:
		/**
		 * Well ... our grid :-)
		 */
		std::vector < std::vector < T > > data_;
		std::vector < int > data_used_;

		/**
		 * Dimension of a cell in each dimension
		 */
		const float cell_size_;

		/**
		 * The number of cells in each dimension; total number of cells == number_of_cells_per_dimension_³
		 */
		const int number_of_cells_per_dimension_;

		mutable cupp::memory1d<T_device_type> *device_data_ptr_;
		mutable cupp::memory1d<int>   *device_index_ptr_;

		const float world_size_;

		mutable std::vector<T_device_type> device_data_;
		mutable std::vector<int>   device_index_;

		const int *const_symbol_memory;
	public:

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		explicit grid( const float world_size, const std::size_t number_of_cells_per_dimension/*, const int* symbol_mem*/ )
		             :
		              data_ ( number_of_cells_per_dimension*number_of_cells_per_dimension*number_of_cells_per_dimension ),
		              data_used_ ( number_of_cells_per_dimension*number_of_cells_per_dimension*number_of_cells_per_dimension ),
		              cell_size_ ( world_size*2 / number_of_cells_per_dimension ),
		              number_of_cells_per_dimension_ ( number_of_cells_per_dimension ),
		              device_data_ptr_(0),
		              device_index_ptr_(0),
		              world_size_(world_size),
		              const_symbol_memory(0)//symbol_mem)
		{}

		/**
		 * This will insert an element into the grid
		 * @param inserted The element that will be inserted
		 */
		void insert ( const T& inserted, const OpenSteer::Vec3 &position ) {
			int where_to_insert  = static_cast <int> ( (position.x+world_size_) / cell_size_);
			where_to_insert     += static_cast <int> ( (position.y+world_size_) / cell_size_) * number_of_cells_per_dimension_;
			where_to_insert     += static_cast <int> ( (position.z+world_size_) / cell_size_) * number_of_cells_per_dimension_ * number_of_cells_per_dimension_;
			
			const int where_2 = data_used_[where_to_insert];
			
			if (where_2+1 == data_[where_to_insert].size()) {
				data_[where_to_insert].resize (data_[where_to_insert].size()*2);
			}
			
			data_[where_to_insert][where_2] = inserted;
			
			++data_used_[where_to_insert];
		}

		/**
		 * Deletes all values from the grid
		 */
		void clear () {
			for (std::size_t i = 0; i<data_.size(); ++i) {
				//data_[i].clear();
				data_used_[i] = 0;
			}
			device_data_.clear();
			device_index_.clear();
		}
		
	public: /*** CUPP FUNCTIONALITY  ***/

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		device_type transform (const cupp::device &d) {
			int index_counter  = 0;
			
			device_index_.resize (data_.size()+1);

			for (std::size_t i=0; i < data_.size(); ++i) {
				device_index_[i]=index_counter;
				
				
				index_counter+=data_used_[i];
			}
			device_index_[data_.size()]=index_counter;

			device_data_.resize(index_counter);
			
			index_counter = 0;
			for (std::size_t i=0; i < data_.size(); ++i) {
				memcpy ( &device_data_[index_counter], &data_[i][0], data_used_[i] );
				
				index_counter+=data_used_[i];
			}
			


			if (device_data_ptr_ == 0 || device_data_ptr_ -> size() != data_.size()) {
				delete device_data_ptr_;
				device_data_ptr_ = new cupp::memory1d<T_device_type>(d, &device_data_[0], device_data_.size());
			} else {
				device_data_ptr_ -> copy_to_device (device_data_.size(), &device_data_[0]);
			}

			if (device_data_ptr_ == 0 || device_data_ptr_ -> size() != data_.size()) {
				delete device_index_ptr_;
				device_index_ptr_ = new cupp::memory1d<int>(d, &device_index_[0], device_index_.size());
			} else {
				device_index_ptr_ -> copy_to_device (device_index_.size(), &device_index_[0]);
			}

			/*if (cudaMemcpyToSymbol((const char*)const_symbol_memory, &device_index_[0], sizeof(int)*device_index_.size()) != cudaSuccess) {
				throw "doo";
			}*/
			
			return device_type::construct(device_data_ptr_->transform(d), device_index_ptr_->transform(d), cell_size_, number_of_cells_per_dimension_, world_size_);
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
			throw "Should never be called";
		}
};

}


#endif

#endif

