#include<opencv2/core/core.hpp>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/core.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

const int KEY_SPACE = 32;
const int KEY_ESC = 27;
const int KEY_Q = 113;

const int EVENT_LBUTTONDOWN = 1;

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;

void detect(IplImage *img, stringstream &fname);

int main(int argc, char** argv)
{
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

  CvCapture *capture;
  IplImage  *frame;
  int input_resize_percent = 100;

  if(argc < 3)
  {
    std::cout << "Usage " << argv[0] << " cascade.xml video.avi" << std::endl;
    return 0;
  }

  if(argc == 4)
  {
    input_resize_percent = atoi(argv[3]);
    std::cout << "Resizing to: " << input_resize_percent << "%" << std::endl;
  }

  cascade = (CvHaarClassifierCascade*) cvLoad(argv[1], 0, 0, 0);
  storage = cvCreateMemStorage(0);
  capture = cvCaptureFromAVI(argv[2]);

  assert(cascade && storage && capture);

  //Set here the size of the samples you want to create
  Size size(320,240);
  IplImage* frame1 = cvQueryFrame(capture);
  int ImageWidth = size.width;
  int ImageHeight = size.height;
  frame = cvCreateImage(cvSize(ImageWidth , ImageHeight), frame1->depth, frame1->nChannels);

  int key = 0;
  int n=0;

  while(1)
  {
    n++;
    frame1 = cvQueryFrame(capture);

    if(!frame1)
      break;

    cvResize(frame1, frame);

    stringstream filename;
    filename << "pos-" << n;

    detect(frame, filename);

    cvShowImage("video", frame);

    key = cvWaitKey(33);

    if(key == KEY_SPACE)
    {
      key = cvWaitKey(0);
      stringstream filename;
      filename << "neg-" << n << ".png";

      vector<int> compression_params;
      compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
      compression_params.push_back(9);

      try {
         imwrite(filename.str(), cvarrToMat(frame), compression_params);
      }
      catch (runtime_error& ex) {
         fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
         return 1;
      }
    }

    if(key == KEY_ESC)
      break;

  }

  cvDestroyAllWindows();
  cvReleaseImage(&frame);
  cvReleaseCapture(&capture);
  cvReleaseHaarClassifierCascade(&cascade);
  cvReleaseMemStorage(&storage);

  return 0;
}

void detect(IplImage *img, stringstream &fname)
{
  CvSize img_size = cvGetSize(img);
  Mat frame = cvarrToMat(img);
  CvSeq *object = cvHaarDetectObjects(
    img,
    cascade,
    storage,
    1.1, //1.1,//1.5, //-------------------SCALE FACTOR
    1, //2        //------------------MIN NEIGHBOURS
    0, //CV_HAAR_DO_CANNY_PRUNING
    cvSize(0,0),//cvSize( 30,30), // ------MINSIZE
    img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
    );
  int count = object->total;
  std::cout << "Total: " << count << " cars detected." << std::endl;

  for(int i = 0 ; i < ( object ? count : 0 ) ; i++)
  {
    CvRect *r = (CvRect*)cvGetSeqElem(object, i);
    fname << i << ".png";
    cout <<fname.str() << endl;
    //Mat roi(img, Rect(r->x, r->y, r->width, r->height ));
    Mat roi=frame(Rect(r->x, r->y, r->width, r->height ));
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try {
       imwrite(fname.str(), roi, compression_params);
    }
    catch (runtime_error& ex) {
       fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
    fname.str("");
    cvRectangle(img,
      cvPoint(r->x, r->y),
      cvPoint(r->x + r->width, r->y + r->height),
      CV_RGB(255, 0, 0), 2, 8, 0);
  }

  cvShowImage("video", img);
}
