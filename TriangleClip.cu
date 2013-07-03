#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "stdlib.h"
#include "stdio.h"
#include "clip.h"
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#define EPS 0.00001
#define N_STATE 11
#define N_INSTR 14
struct polygon
{
	point p[6];
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
float *d_clipped_vert;
float *d_clipped_vert_3d;
int *d_clipped_n_vert;
int _npair;
instructSet *d_state;
unsigned int mem_size_clipped_vert;
unsigned int mem_size_clipped_n_vert;

#if NVCC_ON 
__constant__ instructSet STATE_SET[N_STATE];
#endif

//#if NVCC_ON
//__host__
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

__host__ void GetResultToHost()
{
	cudaError_t error;
	
	float *h_clipped_vert = (float*)malloc(mem_size_clipped_vert);
	error = cudaMemcpy(h_clipped_vert, d_clipped_vert, mem_size_clipped_vert, cudaMemcpyDeviceToHost);
	CheckError(error);

	int *h_clipped_n_vert = (int*)malloc(mem_size_clipped_n_vert);
	error = cudaMemcpy(h_clipped_n_vert, d_clipped_n_vert, mem_size_clipped_n_vert, cudaMemcpyDeviceToHost);
	CheckError(error);
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

    instructSet is = stateInstr[state];
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


//clip_kernel<<<grid, block>>>(d_trgl_s, d_trgl_c, d_pair, _npair, d_clipped_vert, d_clipped_n_vert);
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
	
	for(int i = 0; i < clipped_cnt; i++)
	{
		clipped[idx].p[i].x = clipped_array[i].x;
		clipped[idx].p[i].y = clipped_array[i].y;
	}
	clipped_n[idx] = clipped_cnt;
}


__host__
vector<point> clip_serial(triangle t_s, triangle t_c)
{
    vector<point> clipped;
    trgl ts, tc;
    int i = 0;
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
    CheckError(error);

    error = cudaMalloc((void **) &d_trgl_c, mem_size);
    CheckError(error);

    error = cudaMemcpy(d_trgl_s, trgl_s, mem_size, cudaMemcpyHostToDevice);
    CheckError(error);

    error = cudaMemcpy(d_trgl_c, trgl_c, mem_size, cudaMemcpyHostToDevice);
    CheckError(error);

    unsigned int mem_size_pair = npair * 2 * sizeof(int);

    error = cudaMalloc((void **) &d_pair, mem_size_pair);
    CheckError(error);

    error = cudaMemcpy(d_pair, pair, mem_size_pair, cudaMemcpyHostToDevice);
    CheckError(error);

    //6 point * 2 value(x and y)
    mem_size_clipped_vert = npair * 12 * sizeof(float);

    error = cudaMalloc((void **) &d_clipped_vert, mem_size_clipped_vert);
	CheckError(error);

	mem_size_clipped_n_vert = npair * sizeof(int);
	error = cudaMalloc((void **) &d_clipped_n_vert, mem_size_clipped_n_vert);
	CheckError(error);

	//!!!!!!!!!!!!!!!!!!!!
	//assign space for stateSet and copy to device memory
	unsigned int mem_size_state = N_INSTR * N_STATE * sizeof(bool);
	error = cudaMalloc((void **) &d_state, mem_size_state);
	error = cudaMemcpy(d_state, _stateSet, mem_size_state, cudaMemcpyHostToDevice);
    CheckError(error);


	_npair = npair;

}

__global__ void padding_kernel(float3* clipped_vert_3d, float2* clipped_vert, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	clipped_vert_3d->x = clipped_vert->x;
	clipped_vert_3d->y = clipped_vert->y;
	clipped_vert_3d->z = 0;
}

__global__ void gen_cells_kernel(int* cellArray, int* preSum, int* nVert, int N)
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

__global__ void gen_points_kernel(float3 *points, float2 *clipped_vert, int *preSum, int *nVert, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	int num = nVert[idx];
	int begin = preSum[idx];
	int tmp;
	for(int i = 0; i < num; i++, begin++)
	{
		tmp = 6 * idx + i;
		points[begin].x = clipped_vert[tmp].x;
		points[begin].y = clipped_vert[tmp].y;
		points[begin].z = 0;
	}
}

__host__
void runKernel(float* &points, int* &cells, int &nCells, int &nPts)//triangle *t_s, triangle *t_c, int2 *pair, int npair)//, polygon *clipped, int *clipped_n)
{
	dim3 block(128, 1, 1);
    dim3 grid(ceil((float)_npair / block.x), 1, 1);
	
	clip_kernel<<<grid, block>>>
		((triangle*)d_trgl_s, (triangle*)d_trgl_c, 
		(int2*)d_pair, _npair, 
		(polygon*)d_clipped_vert, d_clipped_n_vert,
		d_state);

	cudaError_t error;
/*	error = cudaMalloc((void **) &d_clipped_vert_3d, mem_size_clipped_vert * 3 / 2);
	CheckError(error);

	padding_kernel<<<grid, block>>>
		(d_clipped_vert_3d, d_clipped_vert, );
		*/
	int* d_preSum;
    error = cudaMalloc((void **) &d_preSum, mem_size_clipped_n_vert);
    CheckError(error);
	//thrust::device_vector<int> d_vec_clipped_n_vert(d_clipped_n_vert, d_clipped_n_vert + _npair);
//	thrust::device_ptr<int> d_ptr_clipped_preSum = thrust::device_malloc<int>(_npair);

	thrust::device_ptr<int> d_ptr_clipped_n_vert(d_clipped_n_vert);
//	thrust::device_vector<int> d_ptr_clipped_n_vert();

	thrust::device_ptr<int> d_ptr_clipped_preSum(d_preSum);
//	d_ptr_clipped_n_vert
	thrust::exclusive_scan(thrust::device, d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + _npair, d_ptr_clipped_preSum);
	

	int last = d_ptr_clipped_preSum[_npair - 1];
	//cout<<last<<endl;

	nPts = d_ptr_clipped_n_vert[_npair - 1] + last;

	///////////points
	float3* d_points;
	unsigned int mem_size_points = nPts * sizeof(float3);
	error = cudaMalloc((void **) &d_points, mem_size_points);
	gen_points_kernel<<<grid, block>>>(d_points, (float2*)d_clipped_vert, d_preSum, d_clipped_n_vert, _npair);
	float3* h_points = (float3*)malloc(mem_size_points);
	error = cudaMemcpy(h_points, d_points, mem_size_points, cudaMemcpyDeviceToHost);

	//	thrust::device_ptr<int> d_ptr_cells(d_cells);
	//for(int i = 0; i < 20; i++)
	//	std::cout << "D[" << i << "] = " << h_points[i].x <<","<< h_points[i].y <<","<< h_points[i].z << std::endl;
	//////////cell
	/*for(int i = 0; i < 30; i++)
		cout<<d_ptr_clipped_n_vert[_npair -1 - i]<<endl;*/
	thrust::device_ptr<int> d_ptr_clipped_n_vert_end = thrust::remove(thrust::device, d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + _npair, 0);
	nCells = d_ptr_clipped_n_vert_end - d_ptr_clipped_n_vert;
	cout<<"_npair:"<<_npair<<endl;
	cout<<"nCells:" << nCells<<endl;
	/*for(int i = 0; i < 10; i++)
		cout<<d_ptr_clipped_n_vert[nCells -1 - i]<<endl;*/

	int* d_preSum_compact;
	unsigned int mem_size_preSum_compact = nCells * sizeof(int);
	error = cudaMalloc((void **) &d_points, mem_size_preSum_compact);
    CheckError(error);
	thrust::device_ptr<int> d_ptr_clipped_preSum_compact(d_preSum_compact);
//	d_ptr_clipped_n_vert
	thrust::exclusive_scan(thrust::device, d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + nCells, d_ptr_clipped_preSum_compact);

	//thrust::device_ptr<int> aa(d_clipped_n_vert);
	//for(int i = 0; i < 20; i++)
	//	cout<< aa[i]<<endl;

	unsigned int mem_size_cells = (nPts + nCells) * sizeof(int);

	int* d_cells;
    error = cudaMalloc((void **) &d_cells, mem_size_cells);
    CheckError(error);
	
	gen_cells_kernel<<<grid, block>>>(d_cells, d_preSum_compact, d_clipped_n_vert, nCells);

	int* h_cells = (int*)malloc(mem_size_cells);
	error = cudaMemcpy(h_cells, d_cells, mem_size_cells, cudaMemcpyDeviceToHost);
	//for(int i = 0; i < 20; i++)
	//	cout<< h_cells[i]<<endl;



	points = (float*)h_points;
	cells = (int*) h_cells;
	//thrust::device_ptr<point> pts = thrust::device_malloc<point>(nPts);

	//GetResultToHost();


	//thrust::device_vector<int> d_vec_clipped_n_vert(d_ptr_clipped_n_vert, d_ptr_clipped_n_vert + _npair);

}