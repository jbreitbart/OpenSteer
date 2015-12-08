/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_dyn_grid_H
#define DS_dyn_grid_H

#if defined(NVCC)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

#include <limits>
#include <functional>

// cupp
#include "cupp/kernel_type_binding.h"
#include "cupp/device_reference.h"
#include "cupp/vector.h"

#include "OpenSteer/Vec3.h"
#include "OpenSteer/PosIndexPair.h"

#include "deviceT/dyn_grid.h"

#include "dyn_grid_node.h"

namespace ds {



/**
 * @class dyn_grid
 * @author Jens Breitbart
 * @version 0.1
 * @date 11.09.2007
 * @platform Host only
 * @brief 
 */

class dyn_grid {
	public: /*** TYPEDEFS  ***/
		typedef ds::deviceT::dyn_grid                  device_type;
		typedef dyn_grid                               host_type;

	private:
		typedef std::vector<OpenSteer::PosIndexPair>::iterator data_iterator;
		
		std::vector<OpenSteer::PosIndexPair> data_;
		std::vector<dyn_grid_node> index_;
		std::vector<dyn_grid_node> leaves_;
		std::vector<dyn_grid_node> area_;

		mutable cupp::memory1d<OpenSteer::PosIndexPair> *data_d_ptr_;
		mutable cupp::memory1d<dyn_grid_node> *leaves_d_ptr_;
		mutable cupp::memory1d<dyn_grid_node> *area_d_ptr_;

	public:

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		explicit dyn_grid() : data_d_ptr_(0), leaves_d_ptr_(0), area_d_ptr_(0)
		{}

		~dyn_grid() {
			delete data_d_ptr_;
			delete leaves_d_ptr_;
			delete area_d_ptr_;
		}

		void resize(const int size) {
			data_.resize(size);
		}
		
		/**
		 * This will insert an element into the grid
		 * @param inserted The element that will be inserted
		 */
		void insert ( const int ind, const OpenSteer::deviceT::Vec3 &position ) {
			data_.push_back( OpenSteer::PosIndexPair::construct(ind, position) );
		}

		void clear() {
			data_.clear();
			index_.clear();
			leaves_.clear();
			area_.clear();
		}



		struct divide : public std::binary_function<OpenSteer::PosIndexPair, OpenSteer::PosIndexPair, bool> {
			
			divide (){}
			
			bool operator() (const OpenSteer::PosIndexPair& i, const OpenSteer::PosIndexPair& j) const {
				const float i_x = i.position().x();
				const float j_x = j.position().x();
				
				return i_x < j_x;
			}
		};
		
		void order(const int threads_per_block) {
			
			order(data_.begin(), data_.end(), threads_per_block,
			      std::numeric_limits<float>::min(), std::numeric_limits<float>::max()
			     );
			

#if 0
			std::sort( data_.begin(), data_.end(), divide() );
			
			for (int i=0; i<data_.size(); i+=threads_per_block) {

				const data_iterator start = data_.begin()+i;
				const data_iterator end   = data_.end();
				const data_iterator nth   = start+threads_per_block < end ? start+threads_per_block : end;
				
				//std::partial_sort( start, nth, end, divide() );
				
				float min = data_[i].position().x();//std::numeric_limits<float>::max();
				float max = data_[i+threads_per_block-1].position().x();//std::numeric_limits<float>::min();
				/*for (data_iterator iter=start; iter < nth; ++iter) {
					const float x = iter -> position().x();
					if (x<min) {
						min = x;
					}
					if (x>max) {
						max = x;
					}
				}*/

				leaves_.push_back( dyn_grid_node(
				                     i,
				                     min, max,
				                     std::distance(start, nth)
				                   )
				                 );
			}
#endif

			//int sum = 0;
			for (std::size_t i = 0; i < leaves_.size(); ++i) {
				find_area(i);
				//sum += area_[i].size;
			}
			//std::cout << (float)sum/size() << " - " << size()*64 - 7680 << " - " << size() << std::endl;
		}

		int size() const {
			return leaves_.size();
		}
		
	private:

		void find_area (const std::size_t i_) {
			const dyn_grid_node &n = leaves_[i_];

			std::size_t left = 0;
			
			for (std::size_t i=0; i<leaves_.size(); ++i) {
				const dyn_grid_node &c = leaves_[i];
				if (n.low_x - c.high_x >= r ) {
					left = i;
				} else {
					break;
				}
			}

			std::size_t right = leaves_.size()-1;
			
			for (int i=leaves_.size()-1; i>=0; --i) {
				const dyn_grid_node &c = leaves_[i];
				if (c.low_x - n.high_x >= r ) {
					right = i;
				} else {
					break;
				}
			}
			//std::cout << left << ", " << right << std::endl;

			const dyn_grid_node &left_block = leaves_[left];
			const dyn_grid_node &right_block = leaves_[right];


			dyn_grid_node returnee (left_block.index,
			                          left_block.low_x, right_block.high_x,
			                          right_block.index+right_block.size - left_block.index//, 0
			                         );

			area_.push_back (returnee);
			

		}


		class vec3_average : public std::unary_function<OpenSteer::PosIndexPair, void> {
			public:
				vec3_average() : num(0) {}

				void operator() (const OpenSteer::PosIndexPair &i) {
					++num;
					sum += i.position();
				}

				OpenSteer::Vec3 result() const {
					return sum/num;
				}
			private:
				unsigned int num;
				OpenSteer::Vec3 sum;
		};
		
		template <int direction>
		struct compare_vec3 : public std::unary_function<OpenSteer::PosIndexPair, bool> {
			float position;
			
			compare_vec3 (const OpenSteer::Vec3& _position) {
				if (direction == 1) {
					position = _position.x;
				}
				if (direction == 2) {
					position = _position.y;
				}
				if (direction == 3) {
					position = _position.z;
				}
			}
			
			bool operator() (const OpenSteer::PosIndexPair& to_comp) const {
				if (direction == 1) {
					return to_comp.position().x() < position;
				}
				if (direction == 2) {
					return to_comp.position().y() < position;
				}
				if (direction == 3) {
					return to_comp.position().z() < position;
				}
			}
		};
		
		void order(const data_iterator begin, const data_iterator end, const int threads_per_block, const float low_x, const float high_x) {
			/*const int new_index = index_.size();
			index_.push_back( dyn_grid_node(
			                       std::distance(data_.begin(), begin),
			                       low_x, high_x,
                                               std::distance(begin, end)
			                      )
			                );
			*/

			/*if (new_index != 0) {
				index_[parent].nodes[id] = new_index;
			}*/
			
			if (std::distance(begin, end) > threads_per_block) {

				// find center of all agents between begin and end
				
				const OpenSteer::Vec3 center = std::for_each(begin, end, vec3_average()).result();
				/*const data_iterator b1 = end-1;
				const OpenSteer::Vec3 center((begin->position().x() + b1->position().x())/2,
				                             (begin->position().y() + b1->position().x())/2,
				                             (begin->position().z() + b1->position().x())/2);*/
				

				// partition the agents relative to the center
				const data_iterator x_divider = std::partition(begin, end, compare_vec3<1>(center) );

				order (begin, x_divider, threads_per_block, low_x, center.x/*, new_index, 1*/);
				order (x_divider, end, threads_per_block, center.x, high_x/*, new_index, 2*/);
				
			} else {
				leaves_.push_back(dyn_grid_node(
			                       std::distance(data_.begin(), begin),
			                       low_x, high_x,
                                               std::distance(begin, end)/*,
                                               parent*/
			                      ));
			}
		}

	public: /*** CUPP FUNCTIONALITY  ***/

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		device_type transform (const cupp::device &d) {

			if (data_d_ptr_ == 0 || data_d_ptr_ -> size() < data_.size()) {
				delete data_d_ptr_;
				data_d_ptr_ = new cupp::memory1d<OpenSteer::PosIndexPair>(d, &data_[0], data_.size());
			} else {
				data_d_ptr_ -> copy_to_device (data_.size(), &data_[0]);
			}

			if (leaves_d_ptr_ == 0 || leaves_d_ptr_ -> size() < leaves_.size()) {
				delete leaves_d_ptr_;
				leaves_d_ptr_ = new cupp::memory1d<dyn_grid_node>(d, &leaves_[0], leaves_.size());
			} else {
				leaves_d_ptr_ -> copy_to_device (leaves_.size(), &leaves_[0]);
			}

			if (area_d_ptr_ == 0 || area_d_ptr_ -> size() < area_.size()) {
				delete area_d_ptr_;
				area_d_ptr_ = new cupp::memory1d<dyn_grid_node>(d, &area_[0], area_.size());
			} else {
				area_d_ptr_ -> copy_to_device (area_.size(), &area_[0]);
			}

			return device_type::construct(data_d_ptr_->transform(d), leaves_d_ptr_->transform(d), area_d_ptr_->transform(d));
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
