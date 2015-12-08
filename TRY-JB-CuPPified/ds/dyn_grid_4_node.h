#ifndef DS_dyn_grid_node_H
#define DS_dyn_grid_node_H



namespace ds {

struct dyn_grid_node {
	typedef ds::dyn_grid_node                  device_type;
	typedef ds::dyn_grid_node                  host_type;

	
	int index;
	int size;
	//int nodes[3];
	
	float    low_x;
	float    high_x;

	#if !defined(NVCC)
	dyn_grid_node (const int _index, const float _low_x, const float _high_x, const int _size/*, const int parent*/) :
	index(_index),
	size(_size),
	low_x(_low_x), high_x(_high_x)
	{
		/*nodes[0]=parent;
		for (int i=1; i<3; ++i) {
			nodes[i]=-1;
		}*/
	}

	dyn_grid_node (const dyn_grid_node& n) : index(n.index), size(n.size), low_x(n.low_x), high_x(n.high_x) {
		/*for (int i=0; i<3; ++i) {
			nodes[i]=n.nodes[i];
		}*/
	}

	dyn_grid_node(){}
	#endif

/*	bool is_leave() const {
		for (int i=1; i<3; ++i) {
			if (nodes[i]!=-1) return false;
		}
		return true;
	}*/
};

}

#endif
