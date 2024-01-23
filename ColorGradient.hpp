/*
 * A C++ class for defining a color gradient and interpolating a bound
 * value to produce a color value.
 *
 * Copyright (C) 2008  Stefán Freyr Stefánsson, Arnar Birgisson
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GRADIENT_H_
#define GRADIENT_H_

#include <vector>

 /**
  * A representation of a color. Each channel is the standard 8-bit.
  */
typedef struct rgb {
    uint8_t red;
    uint8_t green;
    uint8_t blue;

    bool operator==(rgb lhr)
    {
        return this == &lhr;
    }
}rgb;

/**
 * A color instance that represents an invalid color
 */
static rgb INVALID_COLOR = { 0, 0, 0 };

/**
 * A helper class to map a numerical value to a gradient color scale.
 * The gradient appearance is controlled by specifying a list of colors
 * (called stops) which will then be distributed across a specified
 * range of values. Users can get a color value for a specific numerical
 * value and this class will interpolate the correct color value from
 * its stop list.
 */
class ColorGradient
{
private:
    /**
     * The minimum value that this gradient will represent.
     */
    uint16_t m_min;

    /**
     * The color representing the minimum value.
     */
    rgb m_minColor;

    /**
     * The outlier color for min outliers.
     */
    rgb m_minOutlierColor;

    /**
     * The maximum value that this gradient will represent.
     */
    uint16_t m_max;

    /**
     * The color representing the maximum value.
     */
    rgb m_maxColor;

    /**
     * The outlier color for max outliers.
     */
    rgb m_maxOutlierColor;


    /**
     * The color values of the gradient.
     */
    std::vector<rgb> m_stops;

    /**
     * The workhorse of the class. This method calculates the color that
     * represents the interpolation of the two specified color values using
     * the specified normalized value.
     * @param c1  the color representing the normalized value 0.0
     * @param c2  the color representing the normalized value 1.0
     * @param normalized_value  a value between 0.0 and 1.0 representing
     *                          where on the color scale between c1 and c2
     *                          the returned color should be.
     * @return the interpolated color at normalized_value between c1 and c2.
     */
    rgb interpolate(rgb c1, rgb c2, float normalized_value) {
        if (normalized_value <= 0.0) { return c1; }
        if (normalized_value >= 1.0) { return c2; }

        uint8_t red = (uint8_t)((1.0 - normalized_value) * c1.red +
            normalized_value * c2.red);
        uint8_t green = (uint8_t)((1.0 - normalized_value) * c1.green +
            normalized_value * c2.green);
        uint8_t blue = (uint8_t)((1.0 - normalized_value) * c1.blue +
            normalized_value * c2.blue);

        return rgb{ red, green, blue };
    }

public:

    /**
     * Creates a new uninitialized instance of a Gradient. The caller must
     * initialize the instance with values before use by calling the
     * initialize method.
     */
    ColorGradient() {}

    /**
     * Initializes this gradient by defining the range and intermediate stops as well as
     * optional outlier colors (both for values less that the minimum as well as values
     * more than the maximum. If outlier colors are not specified the first and last colors
     * of the stops will be used for all values
     * less than min and greater than max respectively.
     */
    void initialize(uint16_t min, uint16_t max, std::vector<rgb> stops,
        rgb minOutlierColor = INVALID_COLOR, rgb maxOutlierColor = INVALID_COLOR) {
        m_min = min;
        m_max = max;
        m_stops = stops;
        m_minOutlierColor = minOutlierColor;
        m_maxOutlierColor = maxOutlierColor;
    }

    /**
     * Creates a new instance of a Gradient and initializes it with the specified values.
     */
    ColorGradient(uint16_t min, uint16_t max, std::vector<rgb> stops,
        rgb minOutlierColor = INVALID_COLOR, rgb maxOutlierColor = INVALID_COLOR) {
        initialize(min, max, stops, minOutlierColor, maxOutlierColor);
    }

    /**
     * Destructor.
     */
    virtual ~ColorGradient() {}

    /**
     * Retrieve an RGB color struct for a specified value. Caller must have invoked
     * initialize before calling this method.
     */
    rgb getRgb(uint16_t value) {
        // Handle outliers
        if (value < m_min) {
            return m_minOutlierColor ==
                INVALID_COLOR ? m_stops.front() : m_minOutlierColor;
        }
        if (value > m_max) {
            return m_maxOutlierColor ==
                INVALID_COLOR ? m_stops.back() : m_maxOutlierColor;
        }

        // Find the "bin" that value falls in
        uint16_t range = m_max - m_min;
        uint16_t v = value - m_min;
        float step = range / (float)(m_stops.size() - 1);
        int bin = (int)(v / step);

        // Normalize value in the interval (0,1]
        float normalized_v = (v - bin * step) / step;

        return interpolate(m_stops[bin], m_stops[bin + 1], normalized_v);
    }
};

#endif /*GRADIENT_H_*/