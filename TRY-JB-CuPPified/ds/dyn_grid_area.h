#ifndef DS_dyn_grid_area_H
#define DS_dyn_grid_area_H



namespace ds {

struct dyn_grid_area {
	typedef ds::dyn_grid_area                  device_type;
	typedef ds::dyn_grid_area                  host_type;

	
	int index;
	int size;

	#if !defined(NVCC)
	dyn_grid_area (const int _index, const int _size) :
	index(_index),
	size(_size)
	{}

	dyn_grid_area (const dyn_grid_area& n) : index(n.index), size(n.size) {}

	dyn_grid_area(){}
	#endif

};

}

#endif
