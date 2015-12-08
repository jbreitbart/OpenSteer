/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_dyn_grid_clever_v2_H
#define DS_dyn_grid_clever_v2_H

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

#include "deviceT/dyn_grid_clever_v2.h"

#include "dyn_grid_clever_v2_node.h"

namespace ds {



/**
 * @class dyn_grid_clever_v2
 * @author Jens Breitbart
 * @version 0.1
 * @date 01.10.2007
 * @platform Host only
 * @brief 
 */

class dyn_grid_clever_v2 {
	public: /*** TYPEDEFS  ***/
		typedef ds::deviceT::dyn_grid_clever_v2                  device_type;
		typedef dyn_grid_clever_v2                               host_type;

	private:
		/**
		 * Well ... our grid :-)
		 */
		std::vector<OpenSteer::PosIndexPair> data_;
		typedef std::vector<OpenSteer::PosIndexPair>::iterator data_iterator;
		

		std::vector<dyn_grid_clever_v2_node> index_;
		std::vector<dyn_grid_clever_v2_node> leaves_;
		std::vector<dyn_grid_clever_v2_node> area_;

		mutable cupp::memory1d<OpenSteer::PosIndexPair> *data_d_ptr_;
		mutable cupp::memory1d<dyn_grid_clever_v2_node> *leaves_d_ptr_;
		mutable cupp::memory1d<dyn_grid_clever_v2_node> *area_d_ptr_;

	public:

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		explicit dyn_grid_clever_v2() : data_d_ptr_(0), leaves_d_ptr_(0), area_d_ptr_(0)
		{}

		~dyn_grid_clever_v2() {
			delete data_d_ptr_;
			delete leaves_d_ptr_;
			delete area_d_ptr_;
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

		void order(const int threads_per_block) {
			order(data_.begin(), data_.end(), threads_per_block,
			      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
			      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
			      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
			      0, 0
			     );

			for (std::size_t i = 0; i < leaves_.size(); ++i) {
				find_area(i);
			}
			
			/*int sum = 0;
			for (std::size_t i = 0; i < leaves_.size(); ++i) {
				sum += area_[i].size;
			}
			std::cout << "1: " << sum/size() << " - " << size()*64 - 7680 << " - " << size() << std::endl;*/
		}

		int size() const {
			return leaves_.size();
		}
		
	private:


		void find_area (const std::size_t i_) {
		
			const dyn_grid_clever_v2_node &n = leaves_[i_];

			if ( n.high_x - n.low_x < r || n.high_y - n.low_y < r || n.high_z - n.low_z < r) {
				area_.push_back (find_area_top (i_, leaves_[i_].nodes[0]));
			} else {
				area_.push_back (n);
			}
		}



		dyn_grid_clever_v2_node find_area_top (const std::size_t source, const std::size_t current) const {
			const dyn_grid_clever_v2_node &cur_block = leaves_ [source];
			const dyn_grid_clever_v2_node &temp      = index_  [current];
			
			if ( !(cur_block.high_x != temp.high_x && temp.low_x != cur_block.low_x && cur_block.high_y != temp.high_y && temp.low_y != cur_block.low_y && cur_block.high_z != temp.high_z && temp.low_z != cur_block.low_z) && current != 0) {
				return find_area_top (source, temp.nodes[0]);
			}

			return temp;
		}
		
		struct is_valid_left : public std::binary_function <dyn_grid_clever_node, dyn_grid_clever_node, bool> {
			bool operator() (const dyn_grid_clever_node& check, const dyn_grid_clever_node& source) const {
				return ( std::abs(source.high_x - check.low_x) >= r ) &&
				       ( std::abs(source.high_y - check.low_y) >= r ) &&
				       ( std::abs(source.high_z - check.low_z) >= r );
			}
		};

		struct is_valid_right : public std::binary_function <dyn_grid_clever_node, dyn_grid_clever_node, bool> {
			bool operator() (const dyn_grid_clever_node& check, const dyn_grid_clever_node& source) const {
				return ( std::abs(check.high_x - source.low_x) >= r ) &&
				       ( std::abs(check.high_y - source.low_y) >= r ) &&
				       ( std::abs(check.high_z - source.low_z) >= r );
			}
		};


		class vec3_average : public std::unary_function<OpenSteer::PosIndexPair, void> {
			public:
				vec3_average() : num(0) {}

				void operator() (const OpenSteer::PosIndexPair &i) {
					++num;
					sum += i.position();
					//std::cout << i.position().x() << ", "  << i.position().y() << ", " << i.position().z() << std::endl;
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
		
		void order(const data_iterator begin, const data_iterator end, const int threads_per_block, const float low_x, const float high_x, const float low_y, const float high_y, const float low_z, const float high_z, const std::size_t parent, const std::size_t id) {
			
			const int new_index = index_.size();
			index_.push_back( dyn_grid_clever_v2_node(
			                       std::distance(data_.begin(), begin),
			                       low_x, high_x, low_y, high_y, low_z, high_z,
                                               std::distance(begin, end),
                                               parent
			                      )
			                );

			if (new_index != 0) {
				index_[parent].nodes[id] = new_index;
			}
			
			if (std::distance(begin, end) > threads_per_block) {

				// find center of all agents between begin and end
				
				const OpenSteer::Vec3 center = std::for_each(begin, end, vec3_average()).result();

				//std::cout << center << std::endl;

				const float low_x_d = std::abs(center.x - low_x);
				const float high_x_d = std::abs(high_x - center.x);
				const float x_comp = low_x_d > high_x_d ? low_x_d : high_x_d;

				const float low_y_d = std::abs(center.y - low_y);
				const float high_y_d = std::abs(high_y - center.y);
				const float y_comp = low_y_d > high_y_d ? low_y_d : high_y_d;

				const float low_z_d = std::abs(center.z - low_z);
				const float high_z_d = std::abs(high_z - center.z);
				const float z_comp = low_z_d > high_z_d ? low_z_d : high_z_d;

				if (x_comp > y_comp && x_comp > z_comp) {
					//std::cout << "X: " << x_comp << ", " << y_comp << ", " << z_comp << std::endl;
					// partition the agents relative to the center
					const data_iterator x_divider = std::partition(begin, end, compare_vec3<1>(center) );

					order (begin, x_divider, threads_per_block, low_x, center.x, low_y, high_y, low_z, high_z, new_index, 1);
					order (x_divider, end, threads_per_block, center.x, high_x, low_y, high_y, low_z, high_z, new_index, 2);

					return;
				}

				//if (div_direction==2) {
				if (y_comp > x_comp && y_comp > z_comp) {
					//std::cout << "Y: " << x_comp << ", " << y_comp << ", " << z_comp << std::endl;
					// partition the agents relative to the center
					const data_iterator y_divider = std::partition(begin, end, compare_vec3<2>(center) );

					order (begin, y_divider, threads_per_block, low_x, high_x, low_y, center.y, low_z, high_z, new_index, 1);
					order (y_divider, end, threads_per_block, low_x, high_x, center.y, high_y, low_z, high_z, new_index, 2);

					return;
				}

				if (z_comp > x_comp && z_comp > y_comp) {
					//std::cout << "Z: " << x_comp << ", " << y_comp << ", " << z_comp << std::endl;
					const data_iterator z_divider = std::partition(begin, end, compare_vec3<3>(center) );

					order (begin, z_divider, threads_per_block, low_x, high_x, low_y, high_y, low_z, center.z, new_index, 1);
					order (z_divider, end, threads_per_block, low_x, high_x, low_y, high_y, center.z, high_z, new_index, 2);

					return;
				}
				
				static int div_direction = 0;
				++div_direction;
				div_direction %= 3;

				//std::cout << "=: " << x_comp << ", " << y_comp << ", " << z_comp << std::endl;

				if (div_direction ==0) {
					// partition the agents relative to the center
					const data_iterator x_divider = std::partition(begin, end, compare_vec3<1>(center) );

					order (begin, x_divider, threads_per_block, low_x, center.x, low_y, high_y, low_z, high_z, new_index, 1);
					order (x_divider, end, threads_per_block, center.x, high_x, low_y, high_y, low_z, high_z, new_index, 2);

					return;
				}

				if (div_direction==1) {
					// partition the agents relative to the center
					const data_iterator y_divider = std::partition(begin, end, compare_vec3<2>(center) );

					order (begin, y_divider, threads_per_block, low_x, high_x, low_y, center.y, low_z, high_z, new_index, 1);
					order (y_divider, end, threads_per_block, low_x, high_x, center.y, high_y, low_z, high_z, new_index, 2);

					return;
				}

				if (div_direction==2) {
					const data_iterator z_divider = std::partition(begin, end, compare_vec3<3>(center) );

					order (begin, z_divider, threads_per_block, low_x, high_x, low_y, high_y, low_z, center.z, new_index, 1);
					order (z_divider, end, threads_per_block, low_x, high_x, low_y, high_y, center.z, high_z, new_index, 2);

					return;
				}
				
			} else {
				leaves_.push_back(index_[new_index]);
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
				leaves_d_ptr_ = new cupp::memory1d<dyn_grid_clever_v2_node>(d, &leaves_[0], leaves_.size());
			} else {
				leaves_d_ptr_ -> copy_to_device (leaves_.size(), &leaves_[0]);
			}

			if (area_d_ptr_ == 0 || area_d_ptr_ -> size() < area_.size()) {
				delete area_d_ptr_;
				area_d_ptr_ = new cupp::memory1d<dyn_grid_clever_v2_node>(d, &area_[0], area_.size());
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

#if 0
should call this approach an octree :-)
		typedef std::list<OpenSteer::PosIndexPair>::iterator iterator;
		class tree {
			struct node {
				const iterator begin;
				const iterator end;
				const float    low_x;
				const float    high_x;
				const float    low_y;
				const float    high_y;
				const float    low_z;
				const float    high_z;
				std::size_t    size;

				const node* nodes[3]; //top, left, right

				node (const iterator &_begin, const iterator &_end, const float _low_x, const float _high_x, const float _low_y, const float _high_y, const float _low_z, const float _high_z) : begin(_begin), end(_end), low_x(_low_x), high_x(_high_x), low_y(_low_y), high_y(_high_y), low_z(_low_z), high_z(_high_z), size(std::distance(_begin, _end))
				{}
			};

			private:
				const node* root;
				std::lis

			public:
				tree(const float world_size) {
					
				}
		};
#endif




#if 0
		dyn_grid_clever_node find_area_down_left(const std::size_t source, const std::size_t current, const bool left) const {

			const dyn_grid_clever_node &source_block = leaves_ [source];
			const dyn_grid_clever_node &cur_block    = index_  [current];

			//std::cout << source << ", " << current << ": " << std::abs(cur_block.high_x - temp.low_x) << " - " << cur_block.high_x << " - " << temp.low_x << std::endl;
			
			if (( (std::abs(source_block.high_x - cur_block.low_x) >= r ) &&
			      (std::abs(source_block.high_y - cur_block.low_y) >= r ) &&
			      (std::abs(source_block.high_z - cur_block.low_z) >= r )
			    )  && !cur_block.is_leave() ) {
				if (left)
					return find_area_down_left (source, cur_block.nodes[1], false);
				else
					return find_area_down_left (source, cur_block.nodes[2], false);
			}

			const dyn_grid_clever_node &returnee = index_[cur_block.nodes[0]];
			if (( (std::abs(source_block.high_x - returnee.low_x) < r ) ||
			      (std::abs(source_block.high_y - returnee.low_y) < r ) ||
			      (std::abs(source_block.high_z - returnee.low_z) < r )
			    )) {
				//std::cout << "falsch" << std::endl;
			}
			return returnee;
		}

		dyn_grid_clever_node find_area_down_right(const std::size_t source, const std::size_t current, const bool right) const {
			const dyn_grid_clever_node &cur_block = leaves_ [source];
			const dyn_grid_clever_node &temp      = index_  [current];
			
			if (( (std::abs(temp.high_x - cur_block.low_x) >= r) ||
			      (std::abs(temp.high_y - cur_block.low_y) >= r) ||
			      (std::abs(temp.high_z - cur_block.low_z) >= r)
			    ) && !temp.is_leave()) {
				if (right)
					return find_area_down_right (source, temp.nodes[2], false);
				else
					return find_area_down_right (source, temp.nodes[1], false);
			}

			return index_[temp.nodes[0]];
		}


				const data_iterator x_divider = std::partition(begin, end, std::bind2nd(compare_vec3<1>(), center));
				
				const data_iterator y_divider_1 = std::partition(begin, x_divider, std::bind2nd(compare_vec3<2>(), center));
				const data_iterator y_divider_2 = std::partition(x_divider, end, std::bind2nd(compare_vec3<2>(), center));

				const data_iterator z_divider_11 = std::partition(begin, y_divider_1, std::bind2nd(compare_vec3<3>(), center));
				const data_iterator z_divider_12 = std::partition(y_divider_1, x_divider, std::bind2nd(compare_vec3<3>(), center));
				const data_iterator z_divider_21 = std::partition(x_divider, y_divider_2, std::bind2nd(compare_vec3<3>(), center));
				const data_iterator z_divider_22 = std::partition(y_divider_2, end, std::bind2nd(compare_vec3<3>(), center));


				order (begin, z_divider_11, threads_per_block, low_x, center.x, low_y, center.y, low_z, center.z, new_index, 1);
				order (z_divider_11, y_divider_1, threads_per_block, low_x, center.x, low_y, center.y, center.z, high_z, new_index, 2);
				order (y_divider_1, z_divider_12, threads_per_block, low_x, center.x, center.y, high_y, low_z, center.z, new_index, 3);
				order (z_divider_12, x_divider, threads_per_block, low_x, center.x, center.y, high_y, center.z, high_z, new_index, 4);
				order (x_divider, z_divider_21, threads_per_block, center.x, high_x, low_y, center.y, low_z, center.z, new_index, 5);
				order (z_divider_21, y_divider_2, threads_per_block, center.x, high_x, low_y, center.y, center.z, high_z, new_index, 6);
				order (y_divider_2, z_divider_22, threads_per_block, center.x, high_x, center.y, high_y, low_z, center.z, new_index, 7);
				order (z_divider_22, end, threads_per_block, center.x, high_x, center.y, high_y, center.z, high_z, new_index, 8);
#endif

			/*std::cout << "----------------" << std::endl;
			std::cout << leaves_.size() << std::endl;

			for (std::size_t i = 0; i < leaves_.size(); ++i) {
				const node& cur_node = index_[ leaves_[i] ];
				{
					int right = 0;
					for (std::size_t j = i+1; j < leaves_.size(); ++j) {
						
						const node& temp = index_[leaves_[j]];
						
						if ( temp.low_x - cur_node.high_x >= r ) {
							std::cout << i << "r: " << temp.low_x - cur_node.high_x << " - " << right << std::endl;
							break;
						}
						
						++right;
					}
					right_.push_back(right);
				}
				{
					int left = 0;
					for (int j = i-1; j >=0; --j) {
						
						const node& temp = index_[leaves_[j]];
						
						if ( cur_node.low_x - temp.high_x >= r ) {
							std::cout << i << "l: " << cur_node.low_x - temp.high_x << " - " << left << std::endl;
							break;
						}
						
						++left;
					}
					left_.push_back(left);
				}
			}
			for (std::size_t i = 0; i< leaves_.size(); ++i) {
#if 0
				left_.push_back( left_count(i) );
				right_.push_back ( right_count(i) );
#endif
				std::cout << index_[leaves_[i]].get().low_x << " - " << index_[leaves_[i]].get().high_x << " - " << left_[i] << " - " << right_[i] << "- " << left_[i] + right_[i] << std::endl;
			}*/
			
			
			/*int temp = 0;
			for (int i=0; i<leaves_.size(); ++i) {
				temp += index_[leaves_[i]].get().size;
			}
			std::cout << "Auslastung: " << ((double)temp/(double)leaves_.size()) / (double)threads_per_block << std::endl;*/

			//std::cout << "----------------" << std::endl;

		/*int right_count (std::size_t index) const {
			int old_count = -1;
			int count = 0;
			while (count != old_count && index < leaves_.size()) {
				old_count = count;
				count += right_count_ (leaves_[index], leaves_[index], 0, index_[leaves_[index]].high_x, 0);
				index += count+1;
			}
			return count;
		}

		int right_count_ (const std::size_t start, const std::size_t index, int count, const float last_value, const int level) const {
			const node& a = index_[ start ];
			const node& b = index_[index];
			if ( b.low_x - a.high_x < r    &&
			     b.low_x != std::numeric_limits<float>::min()
			   ) {
				if (last_value != b.low_x) {
					count += static_cast<int>(std::pow(2.0, level));
				}

				return right_count_ (start, b.nodes[0], count, b.low_x, level+1);
			}

			return count;
		}
		
		int left_count (std::size_t index) const {
			int old_count = -1;
			int count = 0;
			while (count != old_count && index >=0 ) {
				old_count = count;
				count += left_count_ (leaves_[index], leaves_[index], 0, index_[leaves_[index]].high_x, 0);
				index -= count+1;
			}
			return count;
		}

		int left_count_ (const std::size_t start, const std::size_t index, int count, const float last_value, const int level) const {
			const node& a = index_[ start ];
			const node& b = index_[index];
			if ( std::abs(a.low_x - b.high_x) < r    &&
			     b.high_x != std::numeric_limits<float>::max()
			   ) {
				if (last_value != b.low_x) {
					count += static_cast<int>(std::pow(2.0, level));
				}

				return left_count_ (start, b.nodes[0], count, b.low_x, level+1);
			}

			return count;
		}*/
