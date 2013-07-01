#ifndef CLIP_H
#define CLIP_H

#define NVCC_ON 1	//using nvcc instead of g++

#define EPS 0.00001

//static int cnt = 0;

#if NVCC_ON
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#endif

struct instructSet
{
    bool doIns[14];
};

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
__host__ __device__
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

#if NVCC_ON
__host__ __device__
#endif
//touching boundary is also intersect
inline bool BIntersectIncludeBoundary(pt p1, pt p2, pt q1, pt q2);

#if NVCC_ON
__host__ __device__
#endif
inline bool BIntersect(pt p1, pt p2, pt q1, pt q2);

#if NVCC_ON
__host__ __device__
#endif
inline point diffPt(pt p1, pt p2);

#if NVCC_ON
__host__ __device__
#endif
inline float dot(point p1, point p2);

#if NVCC_ON
__host__ __device__
#endif
inline bool testInside(pt p, trgl t);


#if NVCC_ON
__host__ __device__
#endif
inline void Intersect(pt p1, pt p2, pt q1, pt q2,
        pt &pi, pt &qi);




#if NVCC_ON
__host__ __device__
#endif
inline void AddIntersection(trgl ts, trgl tc, pt *clipped_array, int &clipped_cnt);

#if NVCC_ON
__host__ __device__
#endif
inline void printTrgl(trgl t);

extern "C"
#if NVCC_ON
__host__ __device__
#endif
inline void clip(trgl ts, trgl tc, 	pt clipped_array[6], int &clipped_cnt);

#endif //CLIP_H