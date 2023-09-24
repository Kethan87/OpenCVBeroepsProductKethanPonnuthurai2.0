#include "ShapeFunctions.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	VideoCapture cap(0, cv::CAP_V4L2);

	if (!cap.isOpened())
	{
		cerr << "Error: Could not open camera." << endl;
		return - 1;
	}
	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);

	Mat frame;
	Shape newShape;
	if(argc == 1)
	{
		newShape = initializeNewShape();
        std::thread thread(getInteractiveInput);
        interactiveModus(newShape, frame, cap);
        thread.join();
	} else
	{
		std::thread thread(parser, argv[1]);
		batchModus(frame, cap);
		thread.join();
	}
    return 0;
}
