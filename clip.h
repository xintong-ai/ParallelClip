#ifndef CLIP_H
#define CLIP_H

#include <vector>
#include <vtkType.h>
using namespace std;

#define NVCC_ON 1	//using nvcc instead of g++

#define EPS 0.00001

//static int cnt = 0;

#if NVCC_ON
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#endif


struct pt
{
    float x;
    float y;
    float loc;
#if NVCC_ON
    __host__ __device__
#endif
    pt(float _x, float _y)
    {
        x = _x;
        y = _y;
        loc = -1;
    }
#if NVCC_ON
    __host__ __device__
#endif
	pt()
    {
        loc = -1;
    }
};


struct trgl
{
    //the first 3 points are the vertex
    //others are reserved forintersection points
    pt p[3];
};

struct point
{
    float x;
    float y;
#if NVCC_ON
    __host__ __device__
#endif
    point(float _x, float _y)
    {
        x = _x;
        y = _y;
    }
    point(){};
};


struct triangle
{
    point p[3];
#if NVCC_ON
    __host__ __device__
#endif
    triangle(point p0, point p1, point p2)
    {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
    }
    triangle(point _p[4])
    {
        p[0] = _p[0];
        p[1] = _p[1];
        p[2] = _p[2];
    }
};

#if NVCC_ON
__host__
#endif
void setStateInstr();



template<typename T>
#if NVCC_ON
__host__ __device__
#endif
inline T min3(T x1, T x2, T x3)
{
    T xmin;
    if(x1 < x2)
        xmin = x1;
    else
        xmin = x2;

    if(x2 < xmin)
        xmin = x2;
    if(x3 < xmin)
        xmin = x3;

    return xmin;
}


template<typename T>
#if NVCC_ON
__host__ __device__
#endif
inline T max3(T x1, T x2, T x3)
{
    T xmax;
    if(x1 > x2)
        xmax = x1;
    else
        xmax = x2;

    if(x2 > xmax)
        xmax = x2;
    if(x3 > xmax)
        xmax = x3;

    return xmax;
}


#endif //CLIP_H
__host__
void runKernel(float* &points, vtkIdType* &cells, int &nCells, int &nPts);//triangle *t_s, triangle *t_c, int2 *pair, int npair)//, polygon *clipped, int *clipped_n)

__host__
void loadDataToDevice(float* trgl_s, float* trgl_c, int ntrgl, int *pair, int npair);

__host__
void initCUDA();

__host__
void finishCUDA();

__host__
vector<point> clip_serial(triangle t_s, triangle t_c);

