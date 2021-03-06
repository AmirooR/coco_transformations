#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace std;
using namespace cv;

#define TAG_FLOAT 202021.25

Mat read_flow(const char *filename)
{
    FILE* stream = fopen(filename, "rb");
    int width, height;
    float tag;
    if( (int)fread(&tag, sizeof(float), 1, stream) != 1 ||
        (int)fread(&width, sizeof(int), 1, stream) != 1 ||
        (int)fread(&height, sizeof(int),1, stream) != 1)
    {
        printf("Error in reading width, height, tag\n");
        return Mat();
    }

    if (tag != TAG_FLOAT)
    {
        printf("Error tag is invalid\n");
        return Mat();
    }

    Mat flow(height, width, CV_32FC2);
    for( int r = 0; r < height; r++)
    {
        for(int c = 0; c < width; c++)
        {
            float flow_x, flow_y;
            fread(&flow_x, sizeof(float), 1, stream); //TODO if
            fread(&flow_y, sizeof(float), 1, stream);
            
            flow.at<Point2f>(r, c).x = flow_x;
            flow.at<Point2f>(r, c).y = flow_y;
        }
    }

    fclose(stream);
    return flow;

}

Mat fill_naive_img2(Mat img1, Mat flow)
{
    int* seen = new int[img1.rows*img1.cols];
    float* max_seen_flow_x = new float[img1.rows*img1.cols];
    float* max_seen_flow_y = new float[img1.rows*img1.cols];

    for(int i=0;i<img1.rows*img1.cols; i++) {seen[i] = 0;max_seen_flow_x[i] = 0;max_seen_flow_y[i] = 0;}

    Mat img2_n(img1.rows, img1.cols, CV_8UC3);
    img2_n = Scalar(0, 255, 0);
    for(int r = 0; r < img1.rows; r++)
    {
        for(int c = 0; c < img1.cols; c++)
        {
            Point2f f1 = flow.at<Point2f>(r,c);
            Point2f f;
            f.x = floor(f1.x+c+0.5);
            f.y = floor(f1.y+r+0.5);
            if( f.x >=0 && f.y >=0 && f.x < img1.cols && f.y < img1.rows)
            {
                int idx = (int) f.x + f.y*img1.cols;
                seen[idx]++;
                if(seen[idx] >= 2)
                {
                    float this_flow = abs(f1.x) + abs(f1.y);
                    if( this_flow > abs(max_seen_flow_x[idx]) + abs(max_seen_flow_y[idx]) )
                    {
                        max_seen_flow_x[idx] = f1.x;
                        max_seen_flow_y[idx] = f1.y;
                    }// r,c = f - f1
                    int new_r = (int) f.y - max_seen_flow_y[idx];
                    int new_c = (int) f.x - max_seen_flow_x[idx];
                    img2_n.at<Vec3b>( (int)f.y , (int)f.x) = img1.at<Vec3b>(new_r,new_c);
                }
                else
                {
                    img2_n.at<Vec3b>( (int)f.y , (int)f.x) = img1.at<Vec3b>(r,c);
                    max_seen_flow_x[idx] = f1.x;
                    max_seen_flow_y[idx] = f1.y;
                }
            }
            
        }
    }
    delete[] seen;
    delete[] max_seen_flow_y;
    delete[] max_seen_flow_x;
    return img2_n;
}

int main(int argc, char* argv[])
{
    if( argc != 4)
    {
        printf("Usage: %s img1 img2 flo\n", argv[0]);
        return 0;
    }

    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
    Mat flow = read_flow(argv[3]);
    Mat img2_n = fill_naive_img2(img1, flow);
    imshow("img2_n",img2_n);
    waitKey(0);
    imwrite("img2_n.png", img2_n);
    return 0;
}
