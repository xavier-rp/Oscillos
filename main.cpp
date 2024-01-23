#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>
#include <complex>

#include <fftw3.h>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include "Grid.hpp"
#include "ColorMap.hpp"
#include "CustomAudioStream.hpp"
#include "SamplesRenderer.hpp"
#include "FrequencyRenderer.hpp"

#include "ColorGradient.hpp"

std::vector<float> computeFrequencyAmplitudes(int currentIndex, int nbOfSamples, const sf::Int16* samples, sf::Uint64 sampleCount) {
	double* in1;
	double* in2;
	fftw_complex *out1;
	fftw_complex* out2;
	fftw_plan p1;
	fftw_plan p2;

	in1 = (double*)fftw_malloc(sizeof(double) * nbOfSamples);
	in2 = (double*)fftw_malloc(sizeof(double) * nbOfSamples);
	out1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nbOfSamples);
	out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nbOfSamples);

	int centeringIndex{ 0 };
	for (int i = 0; i < nbOfSamples; i++) {
		centeringIndex = currentIndex;// - nbOfSamples / 2;
		if (centeringIndex < 0) {
			in1[i] = static_cast<double>(0);
			in2[i] = static_cast<double>(0);
		}
		else if (centeringIndex + i >= sampleCount){
			in1[i] = static_cast<double>(0.0);
			in2[i] = static_cast<double>(0.0);
		}
		else {
			in1[i] = static_cast<double>(samples[centeringIndex + 2 * i]);
			in2[i] = static_cast<double>(samples[centeringIndex + 2 * i + 1]);
		}

	}
	p1 = fftw_plan_dft_r2c_1d(nbOfSamples, in1, out1, FFTW_ESTIMATE);

	fftw_execute(p1);

	p2 = fftw_plan_dft_r2c_1d(nbOfSamples, in2, out2, FFTW_ESTIMATE);

	fftw_execute(p2);

	std::vector<float> amplitudes;

	for (int i = 0; i < nbOfSamples / 2; i++) {
		double amplitude1{ sqrt(out1[i][0] * out1[i][0] + out1[i][1] * out1[i][1]) };
		double amplitude2{ sqrt(out2[i][0] * out2[i][0] + out2[i][1] * out2[i][1]) };
		amplitudes.push_back(static_cast<float>((1.0 / nbOfSamples) * (amplitude1 + amplitude2)));
	}

	fftw_destroy_plan(p1);
	fftw_free(in1);
	fftw_free(out1);
	fftw_destroy_plan(p2);
	fftw_free(in2);
	fftw_free(out2);

	return amplitudes;
}

int main()
{
	// Load audio file
	// load an audio buffer from a sound file
	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;
	int FFTLength = 16384;

	sf::SoundBuffer buffer;
	if (!buffer.loadFromFile("audio/lm.wav"))
	{
		std::cout << "Failed to load audio file!" << std::endl;
		return -1;
	}

	const sf::Int16* samples = buffer.getSamples();
	sf::Uint64 sampleCount = buffer.getSampleCount();
	unsigned int sampleRate = buffer.getSampleRate();

	std::vector<float> amplitudes;

	int window_width{ 1900 };
	int window_height{ 1080 };

	Grid grid(static_cast<float>(window_width), static_cast<float>(window_height));

	int nbSamplesToDraw = 44100/4;// 22050;//11025;
	Grid grid2(6400.0f, 540.0f);
	SamplesRenderer samplesRenderer{ buffer, grid2, nbSamplesToDraw};
	Grid grid3(1900.0f, 540.0f);
	FrequencyRenderer frequencyRenderer{ grid3, static_cast<int>(sampleRate) };

	// initialize and play our custom stream
	MyStream playing_stream;
	playing_stream.load(buffer);

	// Create window for visualization
	sf::RenderWindow window(sf::VideoMode(window_width, window_height), "Samples");

	// Start playing the sound
	playing_stream.play();
	sf::Clock globalClock;
	sf::Clock clockSamples;
	sf::Time elapsedSamples = clockSamples.getElapsedTime();
	sf::Clock clockFrequency;
	sf::Time elapsedFrequency = clockFrequency.getElapsedTime();
	int currentSampleIndex{};

	// Main loop
	while (window.isOpen())
	{
		// Handle events
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		// Clear the window
		
		window.clear();

		currentSampleIndex = playing_stream.getCurrentSampleIndex(playing_stream.getPlayingOffset());
		if (elapsedSamples.asSeconds() > 1.0f / 60.0f) {
			samplesRenderer.renderSamples(globalClock.getElapsedTime().asSeconds() * sampleRate);
			clockSamples.restart();
		}
		if (elapsedFrequency.asSeconds() > 1.0f/48.0f) {
			amplitudes = computeFrequencyAmplitudes(globalClock.getElapsedTime().asSeconds() * sampleRate * 2, FFTLength, samples, sampleCount);//(currentSampleIndex, 16384, samples);
			frequencyRenderer.renderFrequencies(amplitudes);
			clockFrequency.restart();
		}

		
		window.clear(sf::Color::Black);
		window.draw(frequencyRenderer.verticesToDraw);
		window.draw(samplesRenderer.chunkToDraw1);
		window.draw(samplesRenderer.chunkToDraw2);

		// Display the window
		window.display();
		elapsedFrequency = clockFrequency.getElapsedTime();
		elapsedSamples = clockSamples.getElapsedTime();

	}

	return 0;
}