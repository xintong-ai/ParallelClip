#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "stdlib.h"
#include "stdio.h"
#include "clip.h"
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#define N_STATE 11
#define N_INSTR 14
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}
 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}

struct polygon
{
	float2 p[6];
};

struct instructSet
{
    bool doIns[N_INSTR];
};

instructSet _stateSet[11];


inline void CheckError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		printf("returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

float *d_trgl_s;
float *d_trgl_c;
int2 *d_pair;
polygon *d_clipped_vert;
int *d_clipped_n_vert;
int _npair;
instructSet *d_state;
unsigned int mem_size_clipped_vert;
unsigned int mem_size_clipped_n_vert;

//#if NVCC_ON 
//__constant__ instructSet STATE_SET[N_STATE];
//#endif


void setStateInstr()
{
    for(int s = 0; s < N_STATE; s++)
        for(int i = 0; i < N_INSTR; i++)
            _stateSet[s].doIns[i] = false;

    _stateSet[0].doIns[1] = true;

    _stateSet[1].doIns[0] = true;
    _stateSet[1].doIns[4] = true;

    _stateSet[2].doIns[1] = true;
    _stateSet[2].doIns[5] = true;

    _stateSet[3].doIns[0] = true;
    _stateSet[3].doIns[4] = true;
    _stateSet[3].doIns[6] = true;

    _stateSet[4].doIns[1] = true;
    _stateSet[4].doIns[5] = true;
    _stateSet[4].doIns[7] = true;

    _stateSet[5].doIns[4] = true;
    _stateSet[5].doIns[6] = true;
    _stateSet[5].doIns[8] = true;

    _stateSet[6].doIns[5] = true;
    _stateSet[6].doIns[7] = true;
    _stateSet[6].doIns[9] = true;

    _stateSet[7].doIns[0] = true;
    _stateSet[7].doIns[12] = true;
    _stateSet[7].doIns[2] = true;
    _stateSet[7].doIns[13] = true;
    _stateSet[7].doIns[4] = true;
    _stateSet[7].doIns[6] = true;

    _stateSet[8].doIns[1] = true;
    _stateSet[8].doIns[12] = true;
    _stateSet[8].doIns[3] = true;
    _stateSet[8].doIns[13] = true;
    _stateSet[8].doIns[5] = true;
    _stateSet[8].doIns[7] = true;

    _stateSet[9].doIns[1] = true;
    _stateSet[9].doIns[5] = true;
    _stateSet[9].doIns[10] = true;
    _stateSet[9].doIns[11] = true;

    _stateSet[10].doIns[1] = true;
    _stateSet[10].doIns[3] = true;
    _stateSet[10].doIns[5] = true;

    //cudaMemcpyToSymbol(STATE_SET,
    //                   &stateSet,
    //                   14 * 11 *sizeof(bool),
    //                   0,
    //                   cudaMemcpyHostToDevice);
}




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
  //shouldn't use EPS for 0 here, otherwise the generated triangle has many holes
  if(tp< - EPS || tp> (1 + EPS) )
      return 0;

  tq = ((p2.y - p1.y)*(q1.x - p1.x) - (p2.x - p1.x)*(q1.y - p1.y))/par;
  //touching the boundary is not inside
  if(tq< - EPS || tq> (1 + EPS))
      return 0;

  return 1;
}



#if NVCC_ON
__host__ __device__
#endif
  //touching the boundary is not inside
inline bool BIntersect(pt p1, pt p2, pt q1, pt q2)
{
  float  tp, tq, par;

  par = (float) ((p2.x - p1.x)*(q2.y - q1.y) -
                 (p2.y - p1.y)*(q2.x - q1.x));

  if (!par) return 0;                               /* parallel lines */
  tp = ((q1.x - p1.x)*(q2.y - q1.y) - (q1.y - p1.y)*(q2.x - q1.x))/par;
	if(tp<EPS || tp> (1 - EPS) )
      return 0;

  tq = ((p2.y - p1.y)*(q1.x - p1.x) - (p2.x - p1.x)*(q1.y - p1.y))/par;

    if(tq<EPS || tq>(1 - EPS))
      return 0;

 // if(tp<=0 || tp>=1 || tq<=0 || tq>=1) return 0;

  return 1;
}

#if NVCC_ON
__host__ __device__
#endif
inline void IntersectIncludeBoundary(pt p1, pt p2, pt q1, pt q2,
        pt &pi, pt &qi)
{
    float tp, tq, par;

    par = (float) ((p2.x - p1.x)*(q2.y - q1.y) -
                   (p2.y - p1.y)*(q2.x - q1.x));

    if (!par)
        return;                               /* parallel lines */

    tp = ((q1.x - p1.x)*(q2.y - q1.y) - (q1.y - p1.y)*(q2.x - q1.x))/par;
    tq = ((p2.y - p1.y)*(q1.x - p1.x) - (p2.x - p1.x)*(q1.y - p1.y))/par;

    if(tp< - EPS || tp>(1 + EPS) || tq< - EPS || tq> (1 + EPS))
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

    if(tp<EPS || tp>(1 - EPS) || tq< EPS || tq> (1 - EPS))
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
	double dot00 = dot(v0, v0);
	double dot01 = dot(v0, v1);
	double dot02 = dot(v0, v2);
	double dot11 = dot(v1, v1);
	double dot12 = dot(v1, v2);

	// Compute barycentric coordinates
	double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	//this EPS has to be very small
	return (u > EPS) && (v > EPS) && (u + v < (1 - EPS));
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
            //Intersect(tc.p[ic], tc.p[(ic+1)%3], ts.p[is], ts.p[(is+1)%3 ],
            //        insect_c, insect_s);
			IntersectIncludeBoundary(tc.p[ic], tc.p[(ic+1)%3], ts.p[is], ts.p[(is+1)%3 ],
                    insect_c, insect_s);

            if(insect_c.loc >= 0)
            {
                insect_c.loc += ic;
                if(clipped_cnt > 0)
                {
					float loc1 = insect_c.loc;
					float loc2 = clipped_array[clipped_cnt - 1].loc;
					//this epsilon could not be too large because loc varies in a small range within [0, 1]
                    if( loc1 - loc2 > EPS2)		
                        clipped_array[clipped_cnt++] = insect_c;
                    else if(loc2 - loc1 > EPS2)
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

//have to use __host__ __device__ here, could not recognize template???
#if NVCC_ON
__host__ __device__
#endif
inline void myswap(pt &a, pt &b)
{
	pt tmp = a;
	a = b;
	b = tmp;
}

#if NVCC_ON
__host__
#endif
inline void printTrgl(trgl t)
{
	cout<<"("<<t.p[0].x << ","<< t.p[1].x << "," << t.p[2].x << "," << t.p[0].x<<endl;
	cout<<"("<<t.p[0].y << ","<< t.p[1].y << "," << t.p[2].y << "," << t.p[0].y<<endl;
}

__host__ void GetResultToHost()
{
	cudaError_t error;
	
	float *h_clipped_vert = (float*)malloc(mem_size_clipped_vert);
	error = cudaMemcpy(h_clipped_vert, d_clipped_vert, mem_size_clipped_vert, cudaMemcpyDeviceToHost);
	CudaSafeCall(error);

	int *h_clipped_n_vert = (int*)malloc(mem_size_clipped_n_vert);
	error = cudaMemcpy(h_clipped_n_vert, d_clipped_n_vert, mem_size_clipped_n_vert, cudaMemcpyDeviceToHost);
	CudaSafeCall(error);
}


#if NVCC_ON
__host__ __device__
#endif
void clip(trgl ts, trgl tc, pt clipped_array[6], int &clipped_cnt, instructSet *stateInstr)
{
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
		if(tc.p[idx].loc == 0 && tc.p[idx + 1].loc == 1)
			myswap(tc.p[idx], tc.p[idx + 1]);
		if(ts.p[idx].loc == 0 && ts.p[idx + 1].loc == 1)
			myswap(ts.p[idx], ts.p[idx + 1]);
	}

	bool test;
	if(1 == cnt_in_c && 1 == cnt_in_s)
		//test = BIntersectIncludeBoundary(ts.p[1], ts.p[2], tc.p[0], tc.p[1]);
		test = BIntersect(ts.p[1], ts.p[2], tc.p[0], tc.p[1]);

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

    instructSet is = stateInstr[state];
	if(is.doIns[0])//+sc
		AddIntersection(tc, ts, clipped_array, clipped_cnt);
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


	//if number of edge less than 3, then this is not a polygon
	if(clipped_cnt > 0 && clipped_cnt < 3)
	{
	//	printTrgl(ts);
	//	printTrgl(tc);
	//	cout<<"state:"<<state<<endl;
	//	cout<<"clipped_cnt:"<<clipped_cnt<<endl;
		//cout<<"state:"<<state<<endl;
		//cout<<"clipped_cnt:"<<clipped_cnt<<endl;
		//cout<<"error:polygon has one or two vertices, impossible case!"<<endl;
		clipped_cnt = 0;
	//	exit(1);
	}

	//clipped_cnt = ts.p[0].x * 1000;//testInside(ts.p[0], tc);
//	clipped_array[0] = ts.p[0];
}


__global__ void clip_kernel(triangle *t_s, triangle *t_c, int2 *pair, int npair, polygon *clipped, int *clipped_n, instructSet *d_state)
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
	clip(ts, tc, clipped_array, clipped_cnt, d_state);
	//if(clipped_cnt > 6)
	//{
	//	clipped_cnt = 7;
	//}
	//
	for(int i = 0; i < clipped_cnt; i++)
	{
		clipped[idx].p[i].x = clipped_array[i].x;
		clipped[idx].p[i].y = clipped_array[i].y;
	}
	//if(clipped_cnt > 6)
	//	asm("trap;");
	clipped_n[idx] = clipped_cnt;
}


__host__
vector<point> clip_serial(triangle t_s, triangle t_c)
{
    vector<point> clipped;
    trgl ts, tc;
    for(int i = 0; i < 3; i++)
    {
        ts.p[i].x = t_s.p[i].x;
        ts.p[i].y = t_s.p[i].y;
        tc.p[i].x = t_c.p[i].x;
        tc.p[i].y = t_c.p[i].y;
    }
	pt clipped_array[6];
	int clipped_cnt = 0;
	clip(ts, tc, clipped_array, clipped_cnt, _stateSet);

    for(int i = 0; i < clipped_cnt; i++)
    {
        point p(clipped_array[i].x, clipped_array[i].y);
        clipped.push_back(p);
    }
    return clipped;
}

__host__ void finishCUDA()
{
	cudaFree(d_clipped_n_vert);
	cudaFree(d_clipped_vert);
	cudaFree(d_trgl_s);
	cudaFree(d_trgl_c);
	cudaFree(d_pair);
	cudaFree(d_state);
}

__host__ void initCUDA()
{
	int devID = 0;

	cudaSetDevice(devID);

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

__host__ void loadDataToDevice(float* trgl_s, float* trgl_c, int ntrgl, int *pair, int npair)
{


    cudaError_t error;
    unsigned int mem_size = ntrgl * 6 * sizeof(float);//3 vertices, each vertex has x and y(2 float)

    error = cudaMalloc((void **) &d_trgl_s, mem_size);
    CudaSafeCall(error);

    error = cudaMalloc((void **) &d_trgl_c, mem_size);
    CudaSafeCall(error);

    error = cudaMemcpy(d_trgl_s, trgl_s, mem_size, cudaMemcpyHostToDevice);
    CudaSafeCall(error);

    error = cudaMemcpy(d_trgl_c, trgl_c, mem_size, cudaMemcpyHostToDevice);
    CudaSafeCall(error);

    unsigned int mem_size_pair = npair * 2 * sizeof(int);

    error = cudaMalloc((void **) &d_pair, mem_size_pair);
    CudaSafeCall(error);

    error = cudaMemcpy(d_pair, pair, mem_size_pair, cudaMemcpyHostToDevice);
    CudaSafeCall(error);

    //6 point * 2 value(x and y)
    mem_size_clipped_vert = npair * sizeof(polygon);

    error = cudaMalloc((void **) &d_clipped_vert, mem_size_clipped_vert);
	CudaSafeCall(error);

	mem_size_clipped_n_vert = npair * sizeof(int);
	error = cudaMalloc((void **) &d_clipped_n_vert, mem_size_clipped_n_vert);
	CudaSafeCall(error);

	//!!!!!!!!!!!!!!!!!!!!
	//assign space for stateSet and copy to device memory
	unsigned int mem_size_state = N_INSTR * N_STATE * sizeof(bool);
	error = cudaMalloc((void **) &d_state, mem_size_state);
	error = cudaMemcpy(d_state, _stateSet, mem_size_state, cudaMemcpyHostToDevice);
    CudaSafeCall(error);


	_npair = npair;

}


__global__ void gen_cells_kernel(vtkIdType* cellArray, int N, int* preSum, int* nVert)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;
	
	int begin = idx + preSum[idx];
	int num = nVert[idx];
	int input = preSum[idx];
	cellArray[begin++] = num;
	for(int i = 0; i < num; i++)
		cellArray[begin++] = input++;
}

__global__ void gen_points_kernel(float3 *points, polygon *clipped_vert, int *preSum, int *nVert, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	int num = nVert[idx];
	int begin = preSum[idx];
	for(int i = 0; i < num; i++, begin++)
	{
		points[begin].x = clipped_vert[idx].p[i].x;
		points[begin].y = clipped_vert[idx].p[i].y;
		points[begin].z = 0;
	}
}

template <typename T>
__host__ void printArray(T *d_array, int size, int num, bool front)
{
	unsigned int mem_size = size * sizeof(T);
	T *h_array;
	h_array = (T*)malloc(mem_size);
	cudaError_t error = cudaMemcpy(h_array, d_array, mem_size, cudaMemcpyDeviceToHost);
    CudaSafeCall(error);
	cout<< "print array:"<<endl;
	for(int i = 0; i < num; i++)
	{
		if(front)
			cout<<h_array[i]<<endl;
		else
			cout<<h_array[size - 1 - i]<<endl;
	}
}

__host__ void printTriangle(triangle* d_trgl, int i)
{
	unsigned int mem_size = sizeof(triangle);
	triangle* h_trgl;
	h_trgl = (triangle*)malloc(mem_size);
	cudaError_t error = cudaMemcpy(h_trgl, d_trgl + i, mem_size, cudaMemcpyDeviceToHost);
    CudaCheckError();
	cout<< "print Triangle:"<<endl;
	for(int j = 0; j < 3; j++)
		cout<<h_trgl->p[j].x<<","<<h_trgl->p[j].y<<endl;
	free(h_trgl);
}



__host__ void printPair(int2 *d_array, int size, int idx)
{
	unsigned int mem_size = size * sizeof(int2);
	int2 *h_array;
	h_array = (int2*)malloc(mem_size);
	cudaError_t error = cudaMemcpy(h_array, d_array, mem_size, cudaMemcpyDeviceToHost);
    CudaSafeCall(error);
	cout<< "print printPair:"<<endl;
	cout<<h_array[idx].x<<","<<h_array[idx].y<<endl;
}

__host__ void printPolygon(polygon *d_array, int size, int idx)
{
	unsigned int mem_size = size * sizeof(polygon);
	polygon *h_array;
	h_array = (polygon*)malloc(mem_size);
	cudaError_t error = cudaMemcpy(h_array, d_array, mem_size, cudaMemcpyDeviceToHost);
    CudaSafeCall(error);
	cout<< "print polygon:"<<endl;
	for(int i = 0; i < 6; i++)
	{
		cout<< h_array[idx].p[i].x << "," << h_array[idx].p[i].y  <<endl;
	}
}



template <typename T>
__host__ void checkArray(T *d_array, int size)
{
	unsigned int mem_size = size * sizeof(T);
	T *h_array;
	h_array = (T*)malloc(mem_size);
	cudaError_t error = cudaMemcpy(h_array, d_array, mem_size, cudaMemcpyDeviceToHost);
    CudaSafeCall(error);
	for(int i = 0; i < size; i++)
	{
		if(h_array[i] > 6)
			cout<<"check:"<<i<<","<<h_array[i]<<endl;
	}
}

__host__
void runKernel(float* &points, vtkIdType* &cells, int &nCells, int &nPts, int nBlock)//triangle *t_s, triangle *t_c, int2 *pair, int npair)//, polygon *clipped, int *clipped_n)
{
	dim3 block(nBlock, 1, 1);
    dim3 grid(ceil((float)_npair / block.x), 1, 1);

	
	//printTriangle((triangle*)d_trgl_s, 16546);
	//printTriangle((triangle*)d_trgl_c, 88008);
	
	clip_kernel<<<grid, block>>>
		((triangle*)d_trgl_s, (triangle*)d_trgl_c, 
		(int2*)d_pair, _npair, 
		d_clipped_vert, d_clipped_n_vert,
		d_state);
	CudaCheckError();



	//printPair(d_pair, _npair, 681046);
	//printPolygon(d_clipped_vert, _npair, 681046);


	//checkArray<int>(d_clipped_n_vert, _npair);

	cudaError_t error;

	int* d_preSum;
    error = cudaMalloc((void **) &d_preSum, mem_size_clipped_n_vert);
    CudaSafeCall(error);

	//previous sum for the number of vertices
	thrust::device_ptr<int> d_ptr_clipped_n_vert(d_clipped_n_vert);
	//cout<<"num of vert:"<<d_ptr_clipped_n_vert[681046]<<endl;
	thrust::device_ptr<int> d_ptr_clipped_preSum(d_preSum);
	thrust::exclusive_scan(thrust::device, d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + _npair, d_ptr_clipped_preSum); 

	nPts = d_ptr_clipped_n_vert[_npair - 1] + d_ptr_clipped_preSum[_npair - 1];
	//cout<<"nPts:"<<nPts<<endl;
	
	///////////points
	float3* d_points;
	unsigned int mem_size_points = nPts * sizeof(float3);
	error = cudaMalloc((void **) &d_points, mem_size_points);
	gen_points_kernel<<<grid, block>>>(d_points, d_clipped_vert, d_preSum, d_clipped_n_vert, _npair);

	float3* h_points = (float3*)malloc(mem_size_points);
	error = cudaMemcpy(h_points, d_points, mem_size_points, cudaMemcpyDeviceToHost);

	//////cells//////
	thrust::device_ptr<int> d_ptr_clipped_n_vert_end = thrust::remove(thrust::device, d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + _npair, 0);
	nCells = d_ptr_clipped_n_vert_end - d_ptr_clipped_n_vert;

	int* d_preSum_compact;
	unsigned int mem_size_preSum_compact = nCells * sizeof(int);
    cudaMalloc((void **) &d_preSum_compact, mem_size_preSum_compact);
	thrust::device_ptr<int> d_ptr_clipped_preSum_compact(d_preSum_compact);
	thrust::exclusive_scan(thrust::device, d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + nCells, d_ptr_clipped_preSum_compact);
	//cout<<"nCells:"<<nCells<<endl;

	
	//cout<<"d_ptr_preSum_compact:"<<endl;
	//for(int i = 0; i < 10; i++)
	//	cout<<d_ptr_clipped_preSum_compact[i]<<endl;

	int size_cells = nPts + nCells;

	unsigned int mem_size_cells = size_cells * sizeof(vtkIdType);


	
	//size_t fr, ttl;
	//cuMemGetInfo(&fr, &ttl);
	//cout<<"fr:"<<fr<<endl;
	//cout<<"ttl:"<<ttl<<endl;

	vtkIdType* d_cells;
    error = cudaMalloc((void **) &d_cells, mem_size_cells);
	CudaSafeCall( error );

	dim3 block2(nBlock, 1, 1);
    dim3 grid2(ceil((float)size_cells / block2.x), 1, 1);
	

	cout<<"grid2:"<<grid2.x<<","<<grid2.y<<","<<grid2.z<<endl;
	gen_cells_kernel<<<grid2, block2>>>(d_cells, nCells, d_preSum_compact, d_clipped_n_vert);
	//printArray<vtkIdType>(d_cells, 100, 10, true);
	//printArray<int>(d_preSum_compact, nCells, 10, false);
	//printArray<int>(d_clipped_n_vert, nCells, 10, false);

	vtkIdType* h_cells = (vtkIdType*)malloc(mem_size_cells);
	error = cudaMemcpy(h_cells, d_cells, mem_size_cells, cudaMemcpyDeviceToHost);
	cudaFree(d_cells);
	cudaFree(d_clipped_n_vert);
	cudaFree(d_preSum);
	cudaFree(d_points);
	
	cudaFree(d_trgl_s);
	cudaFree(d_trgl_c);
	cudaFree(d_pair);
	cudaFree(d_state);
	cudaFree(d_preSum_compact);

	points = (float*)h_points;
	cells = h_cells;
}