#pragma once
#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <SFML/Audio.hpp>
#include "SFML/Graphics.hpp"

#include "ColorMap.hpp"


class FrequencyRenderer {

public:
	Grid& grid;
	int& sampleRate;
	sf::VertexArray verticesToDraw;
	float allTimeMax{ 0 };


	FrequencyRenderer(Grid& grid, int sampleRate) :
		grid{ grid },
		sampleRate{ sampleRate } {

	}

	void renderFrequencies(std::vector<float>& amplitudes) {

		verticesToDraw = sf::VertexArray(sf::LinesStrip, amplitudes.size());

		int binsToCut{ static_cast<int>(amplitudes.size()) / static_cast<int>(grid.width) + 1 };
		float maximumAmplitude{ *std::max_element(amplitudes.begin() + binsToCut, amplitudes.end()) };
		if (maximumAmplitude > allTimeMax) {
			allTimeMax = maximumAmplitude;
		}

		if (maximumAmplitude > 0.0f) {
			for (int i = 0; i < amplitudes.size(); i++) {
				if (i >= binsToCut) {
					//verticesToDraw[i].position = sf::Vector2f(std::log10(static_cast<float>(i) / static_cast<float>(sampleRate / 2) * grid.width) / std::log10(grid.width) * grid.width, -amplitudes[i] + grid.height);// (1.0 - amplitudes[i] / maximumAmplitude)* grid.height);

					// Logscale frequencies
					//verticesToDraw[i].position = sf::Vector2f(std::log10(static_cast<float>(i) / static_cast<float>(amplitudes.size()) * grid.width) / std::log10(grid.width) * grid.width, -std::log10(amplitudes[i] + 1.0f) / std::log10(maximumAmplitude + 1.0f) * grid.height + 2*grid.height);
					verticesToDraw[i].position = sf::Vector2f(std::log10(static_cast<float>(i) / static_cast<float>(amplitudes.size()) * grid.width) / std::log10(grid.width) * grid.width, -std::log10(amplitudes[i] + 1.0f) / std::log10(allTimeMax + 1.0f) * grid.height + 2 * grid.height);

					// Linear frequencies
					//verticesToDraw[i].position = sf::Vector2f(static_cast<float>(i) / static_cast<float>(amplitudes.size()) * grid.width, -std::log10(amplitudes[i] + 1.0f) / std::log10(maximumAmplitude + 1.0f) * grid.height + 2*grid.height);
					verticesToDraw[i].color = sf::Color::Green;
				}
				else {
					verticesToDraw[i].position = sf::Vector2f(0.0f, 2 * grid.height);
					verticesToDraw[i].color = sf::Color::Green;
				}

			}

		}
		else {
			for (int i = 0; i < amplitudes.size(); i++) {
				if (i >= binsToCut) {
					verticesToDraw[i].position = sf::Vector2f(std::log10(static_cast<float>(i) / static_cast<float>(amplitudes.size()) * grid.width) / std::log10(grid.width) * grid.width, 2 * grid.height - 100.0f);

					verticesToDraw[i].color = sf::Color::Green;
				}
				else {
					verticesToDraw[i].position = sf::Vector2f(0.0f, 2 * grid.height - 100.0f);
					verticesToDraw[i].color = sf::Color::Green;
				}

			}

		}


	}
};