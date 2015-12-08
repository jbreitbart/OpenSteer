#ifndef DS_dyn_grid_clever_v2_node_H
#define DS_dyn_grid_clever_v2_node_H



namespace ds {

struct dyn_grid_clever_v2_node {
	typedef ds::dyn_grid_clever_v2_node                  device_type;
	typedef ds::dyn_grid_clever_v2_node                  host_type;

	
	int index;
	int size;
	int nodes[3];
	
	float    low_x;
	float    high_x;
	float    low_y;
	float    high_y;
	float    low_z;
	float    high_z;


	#if !defined(NVCC)
	dyn_grid_clever_v2_node (const int _index, const float _low_x, const float _high_x, const float _low_y, const float _high_y, const float _low_z, const float _high_z, const int _size, const int parent) :
	index(_index),
	size(_size),
	low_x(_low_x), high_x(_high_x), low_y(_low_y), high_y(_high_y), low_z(_low_z), high_z(_high_z)
	{
		nodes[0]=parent;
		for (int i=1; i<3; ++i) {
			nodes[i]=-1;
		}
	}

	dyn_grid_clever_v2_node (const dyn_grid_clever_v2_node& n) : index(n.index), size(n.size), low_x(n.low_x), high_x(n.high_x), low_y(n.low_y), high_y(n.high_y), low_z(n.low_z), high_z(n.high_z) {
		for (int i=0; i<3; ++i) {
			nodes[i]=n.nodes[i];
		}
	}

	dyn_grid_clever_v2_node(){}
	#endif

	bool is_leave() const {
		for (int i=1; i<3; ++i) {
			if (nodes[i]!=-1) return false;
		}
		return true;
	}
};

}

#endif
