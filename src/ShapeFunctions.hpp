/**
 * @author Kethan
 */

#ifndef SRC_SHAPEFUNCTIONS_HPP_
#define SRC_SHAPEFUNCTIONS_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <cmath>
#include <numeric>

using namespace std;
using namespace cv;

/**
 * @struct Shape
 * @brief Represents a shape with a type and a color.
 */
struct Shape
{
    std::string type;  /**< The type of the shape. */
    std::string color; /**< The color of the shape. */
};

/**
 * @brief Calculate the mid-point coordinates and area of a set of contours.
 *
 * This function returns the middle X and Y coordinates and the area of the shape returned in a string
 *
 * @param contours The contours for which to calculate the information.
 * @param area The area of the contours.
 * @return A string containing the mid-point coordinates and area information.
 */
String getMidXandYandAreaInformation(vector<Point> &contours, uint32_t &area);

/**
 * @brief Get the color range for a given shape.
 *
 * This function returns a vector of color range values based on the provided shape color.
 *
 * @param shape The shape for which to get the color range.
 * @return A vector of color range values.
 */
vector<uint16_t> getColor(const Shape& shape);

/**
 * @brief Detect and gives shape markers based on the given modus
 *
 * This function marks the shapes based on the given modus
 *
 * @param modus the modus of the application ('interactive' or 'batch')
 * @param img The input image.
 * @param aContours the contours on every shape.
 * @param aConPoly the polygons of the shape.
 * @param aBoundRect the bounds of the Rectangle outside the shape.
 * @param iterator the iterator on what shape has to draw the contours on it.
 * @param area the area of the shape.
 *
 */
void showingShapeDetection(const string& modus, Mat img, const vector<vector<Point>>& aContours, vector<Point> aConPoly, const Rect& aBoundRect, uint16_t iterator, uint32_t area);

/**
 * @brief Detect and gives output based on the given modus
 *
 * This function detects and analyzes the shapes based on the given modus
 *
 * @param img The input image.
 * @param shape shape The shape to detect.
 * @param mask mask The mask used for color filtering.
 * @param aModus setting the modus of the application which gives different information ('interactive' or 'batch)
 */
void getShapeDetection(Mat& img, Shape& shape, Mat& mask, const String& aModus);

/**
 * @brief Detect and analyze shapes in an image.
 *
 * This function detects and analyzes shapes in an input image and updates the image with visual markers and information.
 *
 * @param img The input image.
 * @param shape The shape to detect.
 * @param mask The mask used for color filtering.
 */
void getShape(Mat &img, Shape &shape, Mat &mask);

/**
 * @Brief Be able to receive the input of the user to change the newShape object
 */
void getInteractiveInput();

/**
 * @brief Create trackbars for adjusting color parameters interactively.
 * This is only possible if you change the boolean variable 'calibration' to true (now it's currently on false)
 */
void colorControls();

/**
 * @brief Get and display information about a detected shape.
 *
 * This function displays information about the detected shape, including its type and color.
 *
 * @param shape The detected shape.
 */
void getShapeInformation(const Shape &shape);

/**
 * @brief Process frames in batch mode.
 *
 * This function processes frames in batch mode, suitable for continuous video processing.
 *
 * @param frame The input frame.
 * @param cap The video capture object.
 */
void batchModus(Mat frame, VideoCapture cap);

/**
 * @brief Calculate the distance between two points.
 *
 * This function calculates the distance between two points specified by their coordinates.
 *
 * @param x1 The x-coordinate of the first point.
 * @param y1 The y-coordinate of the first point.
 * @param x2 The x-coordinate of the second point.
 * @param y2 The y-coordinate of the second point.
 * @return The calculated distance.
 */
int32_t calculateDistance(int32_t x1, int32_t y1, int32_t x2, int32_t y2);

/**
 * @brief Create trackbars for adjusting color parameters interactively.
 *
 * This function creates trackbars for adjusting color parameters interactively using OpenCV's GUI features.
 *
 * @param img The image to display the trackbars on.
 */
void colorControls(Mat img);

/**
 * @brief Run an interactive mode for shape detection.
 *
 * This function allows users to change the shape detection
 *
 * @param shape The current shape configuration.
 * @param frame The input frame.
 * @param cap The video capture object.
 */
void interactiveModus(Shape& shape, Mat frame, VideoCapture cap);


/**
 * @brief Preprocesses an input image frame for shape detection.
 *
 * This function performs preprocessing operations on the input image frame,
 * such as color conversion, thresholding, blurring, and morphological operations.
 * It is typically used in shape detection tasks.
 *
 * @param shape The shape information to be detected and processed.
 * @param frame The input image frame to be preprocessed.
 * @param calibration Set to true if calibration parameters should be used, false otherwise.
 *
 */
void preprocessing(const Shape& shape, const Mat& frame, bool calibration = false);

/**
 * @brief Get user input for shape and color configuration.
 *
 * This function prompts the user to enter shape and color configuration interactively.
 *
 * @return The initialized shape configuration.
 */
Shape initializeNewShape();

/**
 * @brief Parse shape and color configuration from a file.
 *
 * This function reads shape and color configuration from a file and updates the shape configuration accordingly.
 *
 * @param filename The name of the file to parse.
 */
void parser(const std::string& filename);

/**
 * @brief Calculate and display the time tick difference.
 *
 * This function calculates and displays the time tick difference between two time points.
 *
 * @param start The start time tick.
 * @param stop The stop time tick.
 */
void timeTickDifference(const int64& start,const int64& stop);

/**
 *
 * @brief Calculate the area of a triangle
 *
 * This function calculates the area of a triangle
 *
 * @param aConpoly the lines of the shape
 * @return area of the triangle
 */
uint32_t heronsTriangleFormula(vector<Point>& aConpoly);


/**
 * @brief Verify the shape
 *
 * This function verifies if the input of the shape is correct
 *
 * @param type checks the type of the shape.
 * @param color checks the color of the shape.
 * @return boolean if it's the corrected given shape.
 */
bool verifyTypeAndColor(const string& type,const string& color);



#endif /* SRC_SHAPEFUNCTIONS_HPP_ */

