#pragma once
#include <vector>
#include "SFML/Graphics.hpp"

class Grid {
public:
	float width;
	float height;
	float x_min;
	float x_max;
	float y_min;
	float y_max;




	Grid(float width, float height) :
		width{ width },
		height{ height } {

		float half_width{ width / 2 };
		float half_height{ height / 2 };

		x_min = -half_width;
		x_max = half_width;

		y_min = -half_height;
		y_max = half_height;

	}

};