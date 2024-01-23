#pragma once
#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <SFML/Audio.hpp>
#include "SFML/Graphics.hpp"

#include "ColorMap.hpp"


class SamplesRenderer {

public:
	sf::SoundBuffer& buffer;
	Grid& grid;
	float maximumAmplitude;
	int nbSamplesToDraw;
	std::vector<float> samplesChannel1;
	std::vector<float> samplesChannel2;
	sf::VertexArray verticesChannel1;
	sf::VertexArray verticesChannel2;
	sf::VertexArray chunkToDraw1;
	sf::VertexArray chunkToDraw2;

	SamplesRenderer(sf::SoundBuffer& buffer, Grid& grid, int nbSamplesToDraw) :
		buffer{ buffer },
		grid{ grid },
		nbSamplesToDraw{ nbSamplesToDraw } {

		const sf::Int16* samples = buffer.getSamples();
		float maximumAmplitude{ 0 };

		for (int i = 0; i < buffer.getSampleCount(); i++) {
			if (i % 2 == 0) {
				samplesChannel1.push_back(static_cast<float>(samples[i]));
			}
			else {
				samplesChannel2.push_back(static_cast<float>(samples[i]));
			}
			if (maximumAmplitude < static_cast<float>(std::abs(samples[i]))) {
				maximumAmplitude = static_cast<float>(std::abs(samples[i]));
			}
		}

		for (int i = 0; i < samplesChannel1.size(); i++) {
			samplesChannel1[i] = samplesChannel1[i] / maximumAmplitude;
			samplesChannel2[i] = samplesChannel2[i] / maximumAmplitude;
		}

		verticesChannel1 = sf::VertexArray(sf::Points, samplesChannel1.size());
		verticesChannel2 = sf::VertexArray(sf::Points, samplesChannel2.size());

		for (int i = 0; i < samplesChannel1.size(); i++) {
			//std::cout << i << std::endl;
			verticesChannel1[i].position = sf::Vector2f{ 0 , -samplesChannel1[i] * grid.height / 2.0f + grid.height / 2.0f };
			verticesChannel1[i].color = sf::Color::Blue;
			verticesChannel2[i].position = sf::Vector2f{ i * grid.width / static_cast<float>(nbSamplesToDraw), -samplesChannel2[i] * grid.height / 2.0f + grid.height / 2.0f };
			verticesChannel2[i].color = sf::Color::Green;
		}

		chunkToDraw1 = sf::VertexArray(sf::LinesStrip, nbSamplesToDraw);
		chunkToDraw2 = sf::VertexArray(sf::LineStrip, nbSamplesToDraw);
	}

	void renderSamples(int beginningIndex) {
		int end;
		//i* grid.width / static_cast<float>(nbSamplesToDraw)

		if (beginningIndex + nbSamplesToDraw > samplesChannel1.size()) {
			end = samplesChannel1.size();
		}
		else {
			end = (beginningIndex + nbSamplesToDraw);
		}

		for (int i = beginningIndex; i < end; i++) {
			//std::cout << i << std::endl;
			chunkToDraw1[i - beginningIndex] = verticesChannel1[i];
			chunkToDraw1[i - beginningIndex].position.x = (i - beginningIndex) * grid.width / static_cast<float>(nbSamplesToDraw);
			chunkToDraw2[i - beginningIndex] = verticesChannel2[i];
			chunkToDraw2[i - beginningIndex].position.x = (i - beginningIndex) * grid.width / static_cast<float>(nbSamplesToDraw);

		}

	}

};