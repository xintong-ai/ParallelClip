#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "stdlib.h"
#include "stdio.h"
#include "clip.h"
#include <vector>
#include <iostream>
#define EPS 0.00001
using namespace std;

inline void CheckError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		printf("returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
}


#if NVCC_ON 
__constant__ instructSet STATE_SET[11];
#endif

#if NVCC_ON
__device__
#endif
instructSet stateSet[11];

#if NVCC_ON
__host__ __device__
#endif
void setStateInstr()
{
    for(int s = 0; s < 11; s++)
        for(int i = 0; i < 14; i++)
            stateSet[s].doIns[i] = false;

    stateSet[0].doIns[1] = true;

    stateSet[1].doIns[0] = true;
    stateSet[1].doIns[4] = true;

    stateSet[2].doIns[1] = true;
    stateSet[2].doIns[5] = true;

    stateSet[3].doIns[0] = true;
    stateSet[3].doIns[4] = true;
    stateSet[3].doIns[6] = true;

    stateSet[4].doIns[1] = true;
    stateSet[4].doIns[5] = true;
    stateSet[4].doIns[7] = true;

    stateSet[5].doIns[4] = true;
    stateSet[5].doIns[6] = true;
    stateSet[5].doIns[8] = true;

    stateSet[6].doIns[5] = true;
    stateSet[6].doIns[7] = true;
    stateSet[6].doIns[9] = true;

    stateSet[7].doIns[0] = true;
    stateSet[7].doIns[12] = true;
    stateSet[7].doIns[2] = true;
    stateSet[7].doIns[13] = true;
    stateSet[7].doIns[4] = true;
    stateSet[7].doIns[6] = true;

    stateSet[8].doIns[1] = true;
    stateSet[8].doIns[12] = true;
    stateSet[8].doIns[3] = true;
    stateSet[8].doIns[13] = true;
    stateSet[8].doIns[5] = true;
    stateSet[8].doIns[7] = true;

    stateSet[9].doIns[1] = true;
    stateSet[9].doIns[5] = true;
    stateSet[9].doIns[10] = true;
    stateSet[9].doIns[11] = true;

    stateSet[10].doIns[1] = true;
    stateSet[10].doIns[3] = true;
    stateSet[10].doIns[5] = true;

    //cudaMemcpyToSymbol(STATE_SET,
    //                   &stateSet,
    //                   14 * 11 *sizeof(bool),
    //                   0,
    //                   cudaMemcpyHostToDevice);
}


//template<typename T>
//#if NVCC_ON
//__host__ __device__
//#endif
//inline T min3(T x1, T x2, T x3)
//{
//    T xmin;
//    if(x1 < x2)
//        xmin = x1;
//    else
//        xmin = x2;
//
//    if(x2 < xmin)
//        xmin = x2;
//    if(x3 < xmin)
//        xmin = x3;
//
//    return xmin;
//}


//template<typename T>
//#if NVCC_ON
//__host__ __device__
//#endif
//inline T max3(T x1, T x2, T x3)
//{
//    T xmax;
//    if(x1 > x2)
//        xmax = x1;
//    else
//        xmax = x2;
//
//    if(x2 > xmax)
//        xmax = x2;
//    if(x3 > xmax)
//        xmax = x3;
//
//    return xmax;
//}


#if NVCC_ON
__host__ __device__
#endif
//touching boundary is also intersect
inline bool BIntersectIncludeBoundary(pt p1, pt p2, pt q1, pt q2)
{
  float  tp, tq, par;

  par = (float) ((p2.x - p1.x)*(q2.y - q1.y) -
                 (p2.y - p1.y)*(q2.x - q1.x));

  if (!par) return 0;                               /* parallel lines */

  tp = ((q1.x - p1.x)*(q2.y - q1.y) - (q1.y - p1.y)*(q2.x - q1.x))/par;
  if(tp<0 || tp>1 )
      return 0;

  tq = ((p2.y - p1.y)*(q1.x - p1.x) - (p2.x - p1.x)*(q1.y - p1.y))/par;
  //touching the boundary is not inside
  if(tq<0 || tq>1)
      return 0;

  return 1;
}

#if NVCC_ON
__host__ __device__
#endif
inline bool BIntersect(pt p1, pt p2, pt q1, pt q2)
{
  float  tp, tq, par;

  par = (float) ((p2.x - p1.x)*(q2.y - q1.y) -
                 (p2.y - p1.y)*(q2.x - q1.x));

  if (!par) return 0;                               /* parallel lines */
  tp = ((q1.x - p1.x)*(q2.y - q1.y) - (q1.y - p1.y)*(q2.x - q1.x))/par;
  tq = ((p2.y - p1.y)*(q1.x - p1.x) - (p2.x - p1.x)*(q1.y - p1.y))/par;

  //touching the boundary is not inside
  if(tp<=0 || tp>=1 || tq<=0 || tq>=1) return 0;

  return 1;
}

#if NVCC_ON
__host__ __device__
#endif
inline point diffPt(pt p1, pt p2)
{
	point p(p1.x - p2.x, p1.y - p2.y);
	return p;
}

#if NVCC_ON
__host__ __device__
#endif
inline float dot(point p1, point p2)
{
	return p1.x * p2.x + p1.y * p2.y;
}

#if NVCC_ON
__host__ __device__
#endif
inline bool testInside(pt p, trgl t)
{
	// Compute vectors        
	point v0 = diffPt(t.p[2], t.p[0]);//C - A
	point v1 = diffPt(t.p[1], t.p[0]);// B - A
	point v2 =  diffPt(p, t.p[0]); //P - A

	// Compute dot products
	float dot00 = dot(v0, v0);
	float dot01 = dot(v0, v1);
	float dot02 = dot(v0, v2);
	float dot11 = dot(v1, v1);
	float dot12 = dot(v1, v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
}
	/*
inline bool testInside(pt p, trgl t)
{
    //bool inside = false;
    pt left( -999, p.y);//create(0, point->y, 0, 0, 0, 0, 0, 0, 0, 0.);
    pt right( 999, p.y);//create(0, point->y, 0, 0, 0, 0, 0, 0, 0, 0.);
	bool leftIntersect = false;
	bool rightIntersect = false;
    for(int i = 0; i < 3; i++)
    {
		pt p1, p2;
		p1 = t.p[i];
		p2 = t.p[(i+1)%3]; 
        if(BIntersectIncludeBoundary(left, p, p1, p2))
            leftIntersect = true;
		if(BIntersectIncludeBoundary(right, p, p1, p2))
            rightIntersect = true;
    }
    return (leftIntersect && rightIntersect);
}
*/


#if NVCC_ON
__host__ __device__
#endif
inline void Intersect(pt p1, pt p2, pt q1, pt q2,
        pt &pi, pt &qi)
{
    float tp, tq, par;

    par = (float) ((p2.x - p1.x)*(q2.y - q1.y) -
                   (p2.y - p1.y)*(q2.x - q1.x));

    if (!par)
        return;                               /* parallel lines */

    tp = ((q1.x - p1.x)*(q2.y - q1.y) - (q1.y - p1.y)*(q2.x - q1.x))/par;
    tq = ((p2.y - p1.y)*(q1.x - p1.x) - (p2.x - p1.x)*(q1.y - p1.y))/par;

    if(tp<0 || tp>1 || tq<0 || tq>1)
        return;

//    pi.in = true;
//    qi.in = true;
    pi.x = p1.x + tp*(p2.x - p1.x);
    pi.y = p1.y + tp*(p2.y - p1.y);
    qi.x = pi.x;
    qi.y = pi.y;

    //this can be replaced with tp and tq with care
    pi.loc = tp;// dist(p1.x, p1.y, x, y) / dist(p1.x, p1.y, p2.x, p2.y);
    qi.loc = tq;// dist(q1.x, q1.y, x, y) / dist(q1.x, q1.y, q2.x, q2.y);
}




#if NVCC_ON
__host__ __device__
#endif
inline void AddIntersection(trgl ts, trgl tc, pt *clipped_array, int &clipped_cnt)
{
    for(int ic = 0; ic < 3; ic++)
    {
        for(int is = 0; is < 3; is++)
        {
            pt insect_s, insect_c;
            Intersect(tc.p[ic], tc.p[(ic+1)%3], ts.p[is], ts.p[(is+1)%3 ],
                    insect_c, insect_s);

            if(insect_c.loc >= 0)
            {
                insect_c.loc += ic;
                if(clipped_cnt > 0)
                {
                    if(insect_c.loc > clipped_array[clipped_cnt - 1].loc)
                        clipped_array[clipped_cnt++] = insect_c;
                    else if(insect_c.loc < clipped_array[clipped_cnt - 1].loc)
                    {
                        clipped_array[clipped_cnt] = clipped_array[clipped_cnt - 1];
                        clipped_array[clipped_cnt - 1] = insect_c;
                        clipped_cnt++;
                    }
                    //else :insect_c.loc == clipped_vert[isect_cnt - 1].loc
                    //don't add anything
                }
                else
                {
                    clipped_array[0] = insect_c;
                    clipped_cnt++;
                }
            }
        }
    }
}

#if NVCC_ON
__host__ __device__
#endif
inline void printTrgl(trgl t)
{
	cout<<"("<<t.p[0].x << ","<< t.p[1].x << "," << t.p[2].x << "," << t.p[0].x<<endl;
	cout<<"("<<t.p[0].y << ","<< t.p[1].y << "," << t.p[2].y << "," << t.p[0].y<<endl;
}

extern "C"
#if NVCC_ON
__host__ __device__
#endif
void clip(trgl ts, trgl tc, pt clipped_array[6], int &clipped_cnt)
{
	float sx[2], sy[2], cx[2], cy[2];
    sx[0] = min3<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
	cx[1] = max3<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
	if(sx[0] >= cx[1])
	{
		return;
	}

	sy[0] = min3<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
	cy[1] = max3<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
	if(sy[0] >= cy[1])
	{
		return;
	}

	cx[0] = min3<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
	sx[1] = max3<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
	if(cx[0] >= sx[1])
	{
		return;
	}

	cy[0] = min3<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
	sy[1] = max3<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
	if(cy[0] >= sy[1])
	{
		return;
	}


	//mark inside or outside for the triangle vertices
	//and count the number of inside vertices
	int cnt_in_s = 0, cnt_in_c = 0;
	for(int i = 0; i < 3; i++)
	{
		if(tc.p[i].loc = testInside(tc.p[i], ts))
		   cnt_in_c++;

		if(ts.p[i].loc = testInside(ts.p[i], tc))
			cnt_in_s++;
	}

	//make the "in" vertices in the front of the array
	int a[3] = {0, 1, 0};
	for(int i = 0; i < 3; i++)
	{
		int idx = a[i];
		if(!tc.p[idx].loc && tc.p[idx + 1].loc)
			std::swap(tc.p[idx], tc.p[idx + 1]);
		if(!ts.p[idx].loc && ts.p[idx + 1].loc)
			std::swap(ts.p[idx], ts.p[idx + 1]);
	}

	bool test;
	if(1 == cnt_in_c && 1 == cnt_in_s)
		test = BIntersectIncludeBoundary(ts.p[1], ts.p[2], tc.p[0], tc.p[1]);

	int state = -1;
	if(0 == cnt_in_c && 0 == cnt_in_s)
		state = 0;
	else if(0 == cnt_in_c && 1 == cnt_in_s)
		state = 1;
	else if(1 == cnt_in_c && 0 == cnt_in_s)
		state = 2;
	else if(0 == cnt_in_c && 2 == cnt_in_s)
		state = 3;
	else if(2 == cnt_in_c && 0 == cnt_in_s)
		state = 4;
	else if(0 == cnt_in_c && 3 == cnt_in_s)
		state = 5;
	else if(3 == cnt_in_c && 0 == cnt_in_s)
		state = 6;
	else if(1 == cnt_in_c && 2 == cnt_in_s)
		state = 7;
	else if(2 == cnt_in_c && 1 == cnt_in_s)
		state = 8;
	else if(1 == cnt_in_c && 1 == cnt_in_s && !test)
		state = 9;
	else// if(1 == cnt_in_c && 1 == cnt_in_s && !test1) and (1 == cnt_in_c && 1 == cnt_in_s && test1 && test2)
		state = 10;
	//+cs

    instructSet is = stateSet[state];
	if(is.doIns[0])//+sc
		AddIntersection(tc, ts, clipped_array, clipped_cnt);
	int tmp = clipped_cnt;
	if(is.doIns[1])//+cs
		AddIntersection(ts, tc, clipped_array, clipped_cnt);
	if(is.doIns[12])
		clipped_array[clipped_cnt] = clipped_array[clipped_cnt - 1];
	if(is.doIns[2])//+c0-
		clipped_array[clipped_cnt - 1] = tc.p[0];
	if(is.doIns[3])//+s0-
		clipped_array[clipped_cnt - 1] = ts.p[0];
	if(is.doIns[13])
		clipped_cnt++;
	if(is.doIns[4])//+s0
		clipped_array[clipped_cnt++] = ts.p[0];
	if(is.doIns[5])//+c0
		clipped_array[clipped_cnt++] = tc.p[0];
	if(is.doIns[6])//+s1
		clipped_array[clipped_cnt++] = ts.p[1];
	if(is.doIns[7])//+c1
		clipped_array[clipped_cnt++] = tc.p[1];
	if(is.doIns[8])//+s2
		clipped_array[clipped_cnt++] = ts.p[2];
	if(is.doIns[9])//+c2
		clipped_array[clipped_cnt++] = tc.p[2];
	if(is.doIns[10])//+r0
		clipped_array[clipped_cnt++] = clipped_array[0];
	if(is.doIns[11])//+r0_s0
		clipped_array[0] = ts.p[0];
/*	if(clipped_cnt > 6)
	{
		printTrgl(ts);
		printTrgl(tc);
		cout<<"state:"<<state<<endl;
		cout<<"tmp:"<<tmp<<endl;
		cout<<"clipped_cnt:"<<clipped_cnt<<endl;
	}*/
}


extern "C"
__host__ void initCUDA()
{
	int devID = 0;

//	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
//	{
		//devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		cudaSetDevice(devID);
//	}

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
}

extern "C"
__host__ void loadDataToDevice(float* trgl_s, float* trgl_c, int ntrgl, int *pair, int npar)
{
    float *d_trgl_s;
    float *d_trgl_c;
    int2 *d_pair;
    float *d_clipped_vert;
    float *d_clipped_n_vert;

    cudaError_t error;
    unsigned int mem_size = ntrgl * 6 * sizeof(float);//3 vertices, each vertex has x and y(2 float)

    error = cudaMalloc((void **) &d_trgl_s, mem_size);
    CheckError(error);

    error = cudaMalloc((void **) &d_trgl_c, mem_size);
    CheckError(error);

    error = cudaMemcpy(d_trgl_s, trgl_s, mem_size, cudaMemcpyHostToDevice);
    CheckError(error);

    error = cudaMemcpy(d_trgl_c, trgl_c, mem_size, cudaMemcpyHostToDevice);
    CheckError(error);

    unsigned int mem_size_pair = npar * 2 * sizeof(int);

    error = cudaMalloc((void **) &d_pair, mem_size_pair);
    CheckError(error);

    error = cudaMemcpy(d_pair, pair, mem_size_pair, cudaMemcpyHostToDevice);
    CheckError(error);

    //6 point * 2 value(x and y)
    unsigned int mem_size_clipped_vert = npar * 12 * sizeof(float);

    error = cudaMalloc((void **) &d_clipped_vert, mem_size_clipped_vert);
	CheckError(error);

	unsigned int mem_size_clipped_n_vert = npar * sizeof(int);
	error = cudaMalloc((void **) &d_clipped_n_vert, mem_size_clipped_n_vert);
	CheckError(error);

    setStateInstr();

}

struct polygon
{
	point p[6];
};

__global__ void clip_kernel(triangle *t_s, triangle *t_c, int2 *pair, int npair, polygon *clipped, int *clipped_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= npair)
		return;

	triangle *t_s1 = &t_s[pair[idx].x];
	triangle *t_c1 = &t_c[pair[idx].y];

    trgl ts, tc;
    for(int i = 0; i < 3; i++)
    {
        ts.p[i].x = t_s1->p[i].x;
        ts.p[i].y = t_s1->p[i].y;
        tc.p[i].x = t_c1->p[i].x;
        tc.p[i].y = t_c1->p[i].y;
    }

	pt clipped_array[6];
	int clipped_cnt = 0;
	clip(ts, tc, clipped_array, clipped_cnt);
	
	for(int i = 0; i < clipped_cnt; i++)
	{
		clipped[idx].p[i].x = clipped_array[i].x;
		clipped[idx].p[i].y = clipped_array[i].y;
	}
	clipped_n[idx] = clipped_cnt;
}


