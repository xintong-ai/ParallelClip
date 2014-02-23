#ifndef CLIP_H
#define CLIP_H

#include <vector>
#include <vtkType.h>
#include "vtkUnstructuredGrid.h"
#include "vtkCellArray.h"
using namespace std;

#define NVCC_ON 1	//using nvcc instead of g++

#define EPS 1E-7
//#define EPS2 1E-6
#define EPS3 1E-14
//#define EPS4 1E-5
//#define EPS5 1E-4

//static int cnt = 0;
static clock_t _t0;

#if NVCC_ON
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#endif


struct pt
{
	float2 coord;
    float loc;
#if NVCC_ON
    __host__ __device__
#endif
    pt(float _x, float _y)
    {
        coord.x = _x;
        coord.y = _y;
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


struct pt3
{
	float3 coord;
    float loc;
#if NVCC_ON
    __host__ __device__
#endif
    pt3(float _x, float _y, float _z)
    {
        coord.x = _x;
        coord.y = _y;
		coord.z = _z;
        loc = -1;
    }
#if NVCC_ON
    __host__ __device__
#endif
	pt3()
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

struct trgl3
{
    //the first 3 points are the vertex
    //others are reserved forintersection points
    pt3 p[3];
};
//struct point
//{
//    float x;
//    float y;
//#if NVCC_ON
//    __host__ __device__
//#endif
//    point(float _x, float _y)
//    {
//        x = _x;
//        y = _y;
//    }
//    point(){};
//};


struct triangle
{
    float2 p[3];
#if NVCC_ON
    __host__ __device__
#endif
    triangle(float2 p0, float2 p1, float2 p2)
    {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
    }
    triangle(float2 _p[4])
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
/*
__host__
void runKernel(float* &points, vtkIdType* &cells, int &nCells, int &nPts, int nBlock);//triangle *t_s, triangle *t_c, int2 *pair, int npair)//, polygon *clipped, int *clipped_n)
*/

//
//__host__
//void loadDataToDevice(float* trgl_s, float* trgl_c, int ntrgl, int *pair, int npair);

__host__
void initCUDA();

__host__
void finishCUDA();

__host__
vector<float2> clip_serial(triangle t_s, triangle t_c);

//__host__
//void GetPairs(vtkPoints* vtkPts_s, vtkCellArray* vtkCls_s, 
//	vtkPoints* vtkPts_c, vtkCellArray* vtkCls_c);

//__host__ void runCUDA(vtkPoints* vtkPts_s, vtkCellArray* vtkCls_s, vtkPoints* vtkPts_c, vtkCellArray* vtkCls_c);
__host__ void runCUDA(/*vtkPoints* vtkPts_s, vtkCellArray* vtkCls_s, vtkPoints* vtkPts_c, vtkCellArray* vtkCls_c,*/
    const char* filename_subject, const char* filename_constraint, float binStep,
	float* &points, vtkIdType* &cells, int &nCells, int &nPts, int nBlock);

inline void PrintElapsedTime(char* msg)
{
    clock_t t = clock();
    clock_t compute_time = (t - _t0) * 1000 / CLOCKS_PER_SEC;
    _t0 = t;

    cout<<"Took "<< (float)compute_time * 0.001 << " sec to "<< msg << endl;
}