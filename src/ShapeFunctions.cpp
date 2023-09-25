#include "ShapeFunctions.hpp"


using namespace std;
using namespace cv;

const uint8_t ESCAPE_KEY = 27;
const uint16_t MIN_AREA = 1000;
const double MIN_RECTANGLE_AREA_MARGIN = 0.9;
const double MAX_RECTANGLE_AREA_MARGIN = 1.1;
const double MIN_TRIANGLE_AREA_MARGIN = 0.85;
const double MAX_TRIANGLE_AREA_MARGIN = 1.15;
const double MIN_SQUARE_AREA_MARGIN = 0.85;
const double MAX_SQUARE_AREA_MARGIN = 1.15;
const double MIN_CIRCLE_AREA_MARGIN = 0.85;
const double MAX_CIRCLE_AREA_MARGIN = 1.15;
const double MIN_HALF_CIRCLE_AREA_MARGIN = 0.85;
const double MAX_HALF_CIRCLE_AREA_MARGIN = 1.15;

Mat imgHSV;
Mat mask;
Mat imgDil;
Mat imgBlur;
int hmin = 0;
int smin = 0;
int vmin = 0;
int hmax = 179;
int smax = 255;
int vmax = 255;

bool programOn = true;

Shape newShape;
mutex shapeMutex;

String getMidXandYandAreaInformation(vector<Point>& contours, uint32_t& area)
{
	string midXyAndYcoordinatesInString;
	int averageX = 0;
	int averageY = 0;
	for(auto & contour : contours)
	{
		averageX += contour.x;
		averageY += contour.y;
	}
	averageX /= static_cast<int>(contours.size());
	averageY /= static_cast<int>(contours.size());
	midXyAndYcoordinatesInString = "X: " + to_string(averageX) + " Y: " + to_string(averageY) + " Area: " + to_string(area);
	cout << "Coordinates: " << midXyAndYcoordinatesInString << endl;
	return midXyAndYcoordinatesInString;
}

vector<uint16_t> getColor(const Shape& shape)
{
    vector<uint16_t> colorRange;
    if (shape.color == "groen")
    {
        colorRange.assign({37, 31, 50, 83, 255, 255});
    }
    else if (shape.color == "oranje")
    {
        colorRange.assign({0, 147, 115, 19, 255, 255});
    }
    else if (shape.color == "geel")
    {
        colorRange.assign({20, 80, 100, 30, 255, 255});
    }
    else if (shape.color == "roze")
    {
        colorRange.assign({162, 59, 161, 179, 255, 255});
    }
    return colorRange;
}

int32_t calculateDistance(int32_t x1, int32_t y1, int32_t x2, int32_t y2)
{
    return (static_cast<int32_t>(sqrt(pow(abs(x2 - x1), 2) + pow(abs(y2 - y1), 2) * 1)));
}

uint32_t heronsTriangleFormula(vector<Point>& aConpoly)
{
    int32_t semiPerimeter = 0;
    int32_t sumForPeri = 0;
    double totalTriangleAreaSum = 0.0;
    vector<double> distance;

    for(unsigned int j = 0; j < aConpoly.size() - 1; ++j)
    {
        sumForPeri += calculateDistance(aConpoly.at(j).x, aConpoly.at(j).y, aConpoly.at(j + 1).x,aConpoly.at(j + 1).y);
        distance.push_back(calculateDistance(aConpoly.at(j).x, aConpoly.at(j).y,aConpoly.at(j + 1).x, aConpoly.at(j + 1).y));
    }
    distance.push_back(calculateDistance(aConpoly.at(0).x, aConpoly.at(0).y,aConpoly.at(aConpoly.size() - 1).x, aConpoly.at(aConpoly.size() - 1).y));
    sumForPeri += calculateDistance(aConpoly.at(0).x, aConpoly.at(0).y,aConpoly.at(aConpoly.size() - 1).x, aConpoly.at(aConpoly.size() - 1).y);
    semiPerimeter = sumForPeri / 2;
    totalTriangleAreaSum = semiPerimeter;


    totalTriangleAreaSum *= std::accumulate(distance.begin(), distance.end(), 1.0, [&](double acc, double j) {
        return acc * (std::abs(semiPerimeter) - j);
    });
    totalTriangleAreaSum = sqrt(totalTriangleAreaSum);

    return static_cast<uint32_t>(totalTriangleAreaSum);
}

void showingShapeDetection(const string& modus, Mat img, const vector<vector<Point>>& aContours, vector<Point> aConPoly, const Rect& aBoundRect, uint16_t iterator, uint32_t area)
{
	if(modus == "interactive")
	{
        rectangle(img, aBoundRect.tl(), aBoundRect.br(), Scalar(0, 255, 0), 5);
        drawContours(img, aContours, iterator, Scalar(255, 0, 255), 2);
		putText(img, getMidXandYandAreaInformation(aConPoly, area), { aBoundRect.x + (aBoundRect.width/2), aBoundRect.y + (aBoundRect.height / 2)},FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 69, 255), 2);
	} else if(modus == "batch")
	{
        drawContours(img, aContours, iterator, Scalar(255, 0, 255), 2);
        getMidXandYandAreaInformation(aConPoly, area);
	}
}

void getShapeDetection(Mat& img, Shape& shape, Mat& mask, const String& aModus)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (uint16_t i = 0; i < static_cast<uint16_t>(contours.size()); i++)
    {
    	uint32_t area = static_cast<uint16_t>(contourArea(contours[i]));

        vector<vector<Point>> conPoly(contours.size());
        vector<Rect> boundRect(contours.size());
        if (area > MIN_AREA)
        {
            float peri = static_cast<float>(arcLength(contours[i], true));
            approxPolyDP(contours[i], conPoly[i], 0.03 * peri, true);
            boundRect[i] = boundingRect(conPoly[i]);
            int objCor = static_cast<int>(conPoly[i].size());
            if (shape.type == "driehoek")
            {
	            	uint32_t triangleArea = heronsTriangleFormula(conPoly[i]);
	            	float margin = static_cast<float>(triangleArea) / static_cast<float>(area);
					if (margin > MIN_TRIANGLE_AREA_MARGIN && margin < MAX_TRIANGLE_AREA_MARGIN) {
						showingShapeDetection(aModus, img, contours, conPoly[i], boundRect[i], i, area);
					}
            }
            else if (shape.type == "vierkant")
            {
                if (objCor == 4)
                {
                    float aspRatio = 0.0;
                    aspRatio = static_cast<float>(boundRect[i].width) / static_cast<float>(boundRect[i].height);
                    if (aspRatio > MIN_SQUARE_AREA_MARGIN && aspRatio < MAX_SQUARE_AREA_MARGIN)
                    {
                    	showingShapeDetection(aModus, img, contours, conPoly[i], boundRect[i], i, area);
                    }
                }
            }
            else if (shape.type == "rechthoek")
            {
                if (objCor == 4)
                {
                    double aspRatio = 0.0;
                    aspRatio = static_cast<float>(boundRect[i].width) / static_cast<float>(boundRect[i].height);
                    if (!(aspRatio > MIN_RECTANGLE_AREA_MARGIN &&aspRatio < MAX_RECTANGLE_AREA_MARGIN))
                    {
                    	showingShapeDetection(aModus, img, contours, conPoly[i], boundRect[i], i, area);
                    }
                }
            }else if (shape.type == "cirkel")
                {
                    double circleArea = M_PI * std::pow((boundRect[i].width / 2), 2.0);
                    double marginError = circleArea / area;
                    if (marginError > MIN_CIRCLE_AREA_MARGIN && marginError < MAX_CIRCLE_AREA_MARGIN  && objCor >= 8)
                    {
                    	showingShapeDetection(aModus, img, contours, conPoly[i], boundRect[i], i, area);
                    }
              }else if (shape.type == "halve cirkel")
            {
				int32_t longestDistance = 0;
				int32_t lastDistance = calculateDistance(static_cast<int32_t>(conPoly.at(i).at(0).x), static_cast<int32_t>(conPoly.at(i).at(0).y),static_cast<int32_t>(conPoly.at(i).at(conPoly[i].size() - 1).x), static_cast<int32_t>(conPoly.at(i).at(conPoly[i].size() - 1).y));
	        	for(unsigned int j = 0; j < conPoly[i].size() - 1; ++j)
	        	{
	        		int32_t dist = calculateDistance(conPoly.at(i).at(j).x, conPoly.at(i).at(j).y,conPoly.at(i).at(j + 1).x, conPoly.at(i).at(j + 1).y);

	        		if(dist > longestDistance)
	        		{
	        			longestDistance = dist;
	        		}
	        		if(lastDistance > longestDistance)
	        		{
	        			longestDistance = lastDistance;
	        		}
	        	}
				double circleArea = (M_PI * std::pow((longestDistance / 2),2.0)) / 2;
                if (circleArea / static_cast<float>(area) > MIN_HALF_CIRCLE_AREA_MARGIN && circleArea / static_cast<float>(area) < MAX_HALF_CIRCLE_AREA_MARGIN && objCor >= 5)
                {
                	showingShapeDetection(aModus, img, contours, conPoly[i], boundRect[i], i, area);
                }
            }
        }
    }
}


void getShapeInformation(const Shape &shape)
{
    std::cout << "Type: " << shape.type << " Kleur: " << shape.color << std::endl;
}

void preprocessing(const Shape& shape, const Mat& frame, bool calibration)
{
	int morph_size = 1;
	Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * morph_size + 1, 2 * morph_size + 1),
			Point(morph_size, morph_size));
	vector<uint16_t> colors = getColor(shape);
	cvtColor(frame, imgHSV, COLOR_BGR2HSV);
	Scalar lowerColor = Scalar(colors[0], colors[1], colors[2]);
	Scalar upperColor = Scalar(colors[3], colors[4], colors[5]);
	if (calibration) {
		lowerColor = Scalar(hmin, smin, vmin);
		upperColor = Scalar(hmax, smax, vmax);
	}
	inRange(imgHSV, lowerColor, upperColor, mask);
	GaussianBlur(frame, imgBlur, Size(5, 5), 3, 0);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(mask, imgDil, MORPH_OPEN, element, Point(-1, -1), 2);
}


void batchModus(Mat frame, VideoCapture cap)
{
	while(programOn)
	{
        cap.read(frame);

		if (frame.empty()) {
			cerr << "Error: Frame is leeg." << endl;
			break;
		}
		imshow("Webcam", frame);
		cv::waitKey(1);

		int64 start = getTickCount();
		preprocessing(newShape,frame, false);
		getShapeDetection(frame, newShape, mask, "batch");
        int64 stop = getTickCount();
		timeTickDifference(start, stop);
		imshow("Processed Image", frame);
		imshow("Dilation ", imgDil);
	}

}

void colorControls()
{
	namedWindow("Trackbars", WINDOW_NORMAL);
	resizeWindow("Trackbars", 640, 200);

	createTrackbar("Hue Min", "Trackbars", &hmin, 179);
	createTrackbar("Hue Max", "Trackbars", &hmax, 179);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Val Min", "Trackbars", &vmin, 255);
	createTrackbar("Val Max", "Trackbars", &vmax, 255);

}


void getInteractiveInput()
{
    string userInput;

    while (programOn) {
        cout << "Vul de shape en kleur ('exit' om ui de programma te kunnen gaan): " << endl;
        getline(cin, userInput);

        if (userInput == "exit") {
        	cout << "Programma is gestopt!" << endl;
        	programOn = false;
        }

        stringstream ss(userInput);
        string shapeType;
        string shapeColor;
        ss >> shapeType;

        if (shapeType == "halve") {
            string extra;
            ss >> extra; // this will add 'cirkel' to it.
            shapeType += " " + extra;
        }
        ss >> shapeColor;
        if(!verifyTypeAndColor(shapeType, shapeColor))
        {
        	programOn = false;
        }

        {
            lock_guard<mutex> lock(shapeMutex);
            newShape.type = shapeType;
            newShape.color = shapeColor;
        }

    }
}

void parser(const std::string& filename)
{
	std::ifstream stream(filename);
	if(!stream.is_open())
	{
		std::cout << "File couldn't be opened" << std::endl;
	}
	std::string line;
	while(!stream.eof() && programOn)
	{
		getline(stream,line);
		if (line.length() != 0 && line[0] != '#')
		{
			if(line == "exit")
			{
				cout << "Programma is gestopt!" << endl;
				programOn = false;
			}
			stringstream ss(line);
			string shapeType;
			string shapeColor;
			ss >> shapeType;
			if (shapeType == "halve") {
				string extra;
				ss >> extra; // this will add 'cirkel' to it.
				shapeType += " " + extra;
			}
			ss >> shapeColor;

			{
				lock_guard<mutex> lock(shapeMutex);
				newShape.type = shapeType;
				newShape.color = shapeColor;
			}

			getShapeInformation(newShape);
			std::this_thread::sleep_for(std::chrono::seconds(2));
		}
	}
	stream.close();
}


void interactiveModus(Shape& shape, Mat frame, VideoCapture cap)
{
	colorControls();
	while (programOn) {
		cap.read(frame);

		if (frame.empty()) {
			cerr << "Error: Frame is leeg." << endl;
			break;
		}

		imshow("Webcam", frame);
		{
		lock_guard<mutex> lock(shapeMutex);
        shape.color = newShape.color;
        shape.type = newShape.type;
		}

		int64 start = getTickCount();
		preprocessing(shape,frame,false);
		getShapeDetection(frame, newShape, mask, "interactive");
        int64 stop = getTickCount();
		timeTickDifference(start, stop);
		imshow("Processed Image", frame);
		imshow("Dilation ", imgDil);


		int key = waitKey(1);


		if (key == ESCAPE_KEY) {
			break;
		}
	}

	cap.release();
	destroyAllWindows();
}

Shape initializeNewShape()
{
	string userInput;
	cout << "Vul de shape en kleur ('exit' om ui de programma te kunnen gaan): " << endl;
    getline(cin, userInput);

    stringstream ss(userInput);
    string shapeType;
    string shapeColor;
    ss >> shapeType;

    if (shapeType == "halve") {
        string extra;
        ss >> extra;
        shapeType = shapeType + " " + extra;
    }
    ss >> shapeColor;

    if(!verifyTypeAndColor(shapeType, shapeColor))
    {
    	programOn = false;
    }


	newShape.type = shapeType;
	newShape.color = shapeColor;
	return newShape;

}

bool verifyTypeAndColor(const string& type,const string& color)
{
	if(type == "halve cirkel" || type == "vierkant" || type == "rechthoek" || type == "cirkel" || type == "driehoek")
	{
		if(color == "roze" || color == "geel" || color == "groen" || color == "oranje")
		{
			return true;
		}
	}
	cout << "ERROR!!!: gebruik de volgende shapes om te detecteren: 'driehoek', 'rechthoek', 'halve cirkel', 'vierkant', 'cirkel' " << endl;
	cout << "ERROR!!!: Gebruik de volgende kleuren om de kleur detectie te bepalen: 'roze', 'groen', 'geel', 'oranje'" << endl;
	cout << "SYNTAX: [SHAPE][SPATIE][KLEUR] VOORBEELD: 'halve cirkel groen'" << endl;
	cout << "Start het programma opnieuw op" << endl;
	return false;
}

void timeTickDifference(const int64& start,const int64& stop)
{
	cout << "Duratie van tikken om gedetecteerd te worden: " << (stop - start) << endl;
}








