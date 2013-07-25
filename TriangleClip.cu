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
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridReader.h"
#include <vtkSmartPointer.h>

//#include <math.h>
//#include <math_constants.h>
//#include <math_functions.h>





#define N_STATE 11
#define N_INSTR 14
#define CUDA_ERROR_CHECK
#define RADIUS 1

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

#define M_PI_180 0.01745329252f
#define M_180_PI 57.29577951f
#define M_PI       3.14159265358979323846
#define M_PI_4     0.785398163397448309616
#define M_PI_2     1.57079632679489661923

#define BIN_STEP_X 0.02		//radian
#define BIN_STEP_Y 0.02		//radian

 
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECKcou
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}

inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
//
//inline float rsqrtf(float x)
//{
//    return 1.0f / sqrtf(x);
//}

inline __host__ __device__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}


inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}


inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
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
inline bool BIntersectIncludeBoundary(float2 p1, float2 p2, float2 q1, float2 q2)
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
inline bool on_arc(float3 p1, float3 p2, float3 q)
{
	if(length(p1 - q) < EPS || length(p2 - q) < EPS)
		return true;
	else
		return acos(dot(p1, q)) + acos(dot(p2, q)) - acos(dot(p2, p1)) < EPS3;
}

/*
#if NVCC_ON
__host__ __device__
#endif
inline bool BIntersect(float3 p1, float3 p2, float3 q1, float3 q2)
{
	float3 n1 = cross(p1, p2);
	float3 n2 = cross(q1, q2);
	n1 = normalize(n1);
	n2 = normalize(n2);
	if(length(n1 - n2) < EPS3)
		return false;
	else
	{
		float3 L = normalize(cross(n1, n2));
		float3 L_opp = -L;
		return ((on_arc(p1, p2, L) && on_arc(q1, q2, L)) ||
			(on_arc(p1, p2, L_opp) && on_arc(q1, q2, L_opp) ));
	}
}
*/

#if NVCC_ON
__host__ __device__
#endif
inline float3 BigArcIntersectPoint(float3 p1, float3 p2, float3 q1, float3 q2)
{
    float A = p1.y*p2.z-p1.z*p2.y;
    float B = p1.z*p2.x-p1.x*p2.z;
    float C1 = p1.x*p2.y-p1.y*p2.x;
    float D = q1.y*q2.z-q1.z*q2.y;
    float E = q1.z*q2.x-q1.x*q2.z;
    float F = q1.x*q2.y-q1.y*q2.x;

    float BF_CE = B*F - C1 * E;
    float AE_BD = A*E - B*D;
    float AF_CD = A*F - C1 * D;
    float BD_AE = B*D - A*E;

    float3 L = make_float3(BF_CE / AE_BD, AF_CD/ BD_AE, 1.0f);
    return normalize(L);
}

#if NVCC_ON
__host__ __device__
#endif
inline bool onSegment(float3 p1, float3 p2, float3 q1, float3 q2, float3 p1_p2, float3 q1_q2, float3 &p)
{

    float3 p1_p = cross(p1, p);
    float3 p_p2 = cross(p, p2);


    float3 q1_p = cross(q1, p);
    float3 p_q2 = cross(p, q2);


    float d1 = dot(p1_p, p_p2);
    float d2 = dot(p_p2, p1_p2);
    float d3 = dot(q1_p, p_q2);
    float d4 = dot(p_q2, q1_q2);

    return (d1 > EPS3) &&  (d2 > EPS3)
            && (d3 > EPS3) && (d4 > EPS3);
}

//
#if NVCC_ON
__host__ __device__
#endif
inline bool IntersectCore(float3 p1, float3 p2, float3 q1, float3 q2, float3 &p)// &p = make_float3(0,0,0))
{
    //http://mathforum.org/library/drmath/view/62205.html
	//http://www.boeing-727.com/Data/fly%20odds/distance.html
	//  P . (P1 x P2) = 0

    float A = p1.y*p2.z-p1.z*p2.y;
    float B = p1.z*p2.x-p1.x*p2.z;
    float C1 = p1.x*p2.y-p1.y*p2.x;
    float D = q1.y*q2.z-q1.z*q2.y;
    float E = q1.z*q2.x-q1.x*q2.z;
    float F = q1.x*q2.y-q1.y*q2.x;

    float BF_CE = B*F - C1 * E;
    float AE_BD = A*E - B*D;
    float AF_CD = A*F - C1 * D;
    float BD_AE = -AE_BD;

    float3 p1_p2 =  make_float3(A, B, C1);
    float3 q1_q2 = make_float3(D, E, F);
	p =  make_float3(-BF_CE  , AF_CD , -AE_BD);
	
    float len = length(p);

	if(len < EPS3)
		return false;

    p = p / len;

	if(onSegment(p1, p2, q1, q2, p1_p2, q1_q2, p))
        return true;
    p =  -p;
    if(onSegment(p1, p2, q1, q2, p1_p2, q1_q2, p))
        return true;
    return false;
}

#if NVCC_ON
__host__ __device__
#endif
inline bool BIntersect(float3 p1, float3 p2, float3 q1, float3 q2)//, float3 &p = make_float3(0,0,0))
{
    float3 p;
    return IntersectCore(p1, p2, q1, q2, p);

}

#if NVCC_ON
__host__ __device__
#endif
inline void Intersect(float3 p1, float3 p2, float3 q1, float3 q2,
    pt3 &pi)
{
    float3 interPt;
    bool bInter = IntersectCore(p1, p2, q1, q2, interPt);
    if(bInter)
    {
        pi.coord = interPt;
        pi.loc = length(p1 - interPt) / length(p2 - p1);
    }
    return;
}
	/*
inline bool BIntersect(float3 p1, float3 p2, float3 q1, float3 q2)
{
    float3 L = BigArcIntersectPoint(p1, p2, q1, q2);
    float3 L_opp = -L;
    return ((on_arc(p1, p2, L) && on_arc(q1, q2, L)) ||
        (on_arc(p1, p2, L_opp) && on_arc(p1, p2, L_opp) ));
}

#if NVCC_ON
__host__ __device__
#endif
inline void Intersect(float3 p1, float3 p2, float3 q1, float3 q2,
    pt3 &pi)
{
    float3 L = BigArcIntersectPoint(p1, p2, q1, q2);

    float3 L_opp = -L;
    if(on_arc(p1, p2, L) && on_arc(q1, q2, L))
    {
        pi.coord = L;
        pi.loc = length(p1 - L) / length(p2 - p1);
    }
    else if(on_arc(p1, p2, L_opp) && on_arc(q1, q2, L_opp))
    {
        pi.coord = L_opp;
        pi.loc = length(p1 - L_opp) / length(p2 - p1);
    }
    return;
}
*/

#if NVCC_ON
__host__ __device__
#endif
  //touching the boundary is not inside
inline bool BIntersect(float2 p1, float2 p2, float2 q1, float2 q2)
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
inline void IntersectIncludeBoundary(float2 p1, float2 p2, float2 q1, float2 q2,
        pt &pi)
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
	pi.coord.x = p1.x + tp*(p2.x - p1.x);
    pi.coord.y = p1.y + tp*(p2.y - p1.y);
    //qi.x = pi.x;
    //qi.y = pi.y;

    //this can be replaced with tp and tq with care
    pi.loc = tp;// dist(p1.x, p1.y, x, y) / dist(p1.x, p1.y, p2.x, p2.y);
    //qi.loc = tq;// dist(q1.x, q1.y, x, y) / dist(q1.x, q1.y, q2.x, q2.y);
}

#if NVCC_ON
__host__ __device__
#endif
inline void Intersect(float2 p1, float2 p2, float2 q1, float2 q2,
        pt &pi)
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
	pi.coord.x = p1.x + tp*(p2.x - p1.x);
    pi.coord.y = p1.y + tp*(p2.y - p1.y);
    //qi.x = pi.x;
    //qi.y = pi.y;

    //this can be replaced with tp and tq with care
    pi.loc = tp;// dist(p1.x, p1.y, x, y) / dist(p1.x, p1.y, p2.x, p2.y);
    //qi.loc = tq;// dist(q1.x, q1.y, x, y) / dist(q1.x, q1.y, q2.x, q2.y);
}
/*
#if NVCC_ON
__host__ __device__
#endif
inline void Intersect(float3 p1, float3 p2, float3 q1, float3 q2,
	pt3 &pi)
{
	float3 n1 = cross(p1, p2);
	float3 n2 = cross(q1, q2);
	n1 = normalize(n1);
	n2 = normalize(n2);
	if(length(n1 - n2) >= EPS3)
	{
		float3 L = normalize(cross(n1, n2));
		float3 L_opp = -L;
		if(on_arc(p1, p2, L) && on_arc(q1, q2, L))
		{
			pi.coord = L;
			pi.loc = length(p1 - L) / length(p2 - p1);
		}
		else if(on_arc(p1, p2, L_opp) && on_arc(q1, q2, L_opp))
		{
			pi.coord = L_opp;
			pi.loc = length(p1 - L_opp) / length(p2 - p1);
		}
	}
	return;
}
*/

//
//#if NVCC_ON
//__host__ __device__
//#endif
//inline point diffPt(pt p1, pt p2)
//{
//	point p(p1.x - p2.x, p1.y - p2.y);
//	return p;
//}

//#if NVCC_ON
//__host__ __device__
//#endif
//inline float dot(float2 p1, float2 p2)
//{
//	return p1.x * p2.x + p1.y * p2.y;
//}

//http://forum.beyond3d.com/archive/index.php/t-48658.html
//a, b, c = triangle vertices (in clockwise order)
//x = point on sphere
//
//p1 = dot(x, cross(a, a-c))
//p2 = dot(x, cross(b, b-a))
//p3 = dot(x, cross(c, c-b))

#if NVCC_ON
__host__ __device__
#endif
inline bool testInside(pt p, trgl t)
{
	// Compute vectors        
	float2 v0 = t.p[2].coord - t.p[0].coord;//C - A
	float2 v1 = t.p[1].coord - t.p[0].coord;// B - A
	float2 v2 =  p.coord - t.p[0].coord; //P - A

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
	//this EPS has to be very small
	return (u > EPS) && (v > EPS) && (u + v < (1 - EPS));
}


#if NVCC_ON
__host__ __device__
#endif
inline bool testInside(float3 p, trgl3 t)
{  
	/*
	float3 n[3], next, v1, v2;
	for(int i = 0; i < 3; i++)
	{
		next = t.p[(i + 1) % 3].coord;
		v1 = next - p;
		v2 = next - t.p[i].coord;
		n[i] = cross(v1, v2);
		n[i] = normalize(n[i]);
	}

	for(int i = 0; i < 3; i++)
	{
		if(dot(n[i], n[i + 1]) < EPS3)
			return false;
	}
	return true;
	*/
	float3 n[3], e[3];
    float d[3];
    bool b[3];
    for(int i = 0; i < 3; i++)
        n[i] = cross(t.p[(i + 1) % 3].coord ,t.p[(i + 1) % 3].coord - t.p[i].coord);

    for(int i = 0; i < 3; i++)
    {
        d[i] = dot(n[i], p);
        b[i] = d[i] > EPS3;
    }

    if((b[0] && b[1] && b[2]) || (!b[0] && !b[1] && !b[2]))
        return true;

    return false;
}

#if NVCC_ON
__host__ __device__
#endif
inline void AddIntersection(trgl3 ts, trgl3 tc, pt3 *clipped_array, int &clipped_cnt)
{
    for(int ic = 0; ic < 3; ic++)
    {
        for(int is = 0; is < 3; is++)
        {
            pt3 insect_c;
            //Intersect(tc.p[ic], tc.p[(ic+1)%3], ts.p[is], ts.p[(is+1)%3 ],
            //        insect_c, insect_s);
			/*IntersectIncludeBoundary(tc.p[ic], tc.p[(ic+1)%3], ts.p[is], ts.p[(is+1)%3 ],
                    insect_c, insect_s);*/
			Intersect(tc.p[ic].coord, tc.p[(ic+1)%3].coord, ts.p[is].coord, ts.p[(is+1)%3 ].coord, insect_c);

            if(insect_c.loc >= 0)
            {
                insect_c.loc += ic;
                if(clipped_cnt > 0)
                {
					float loc1 = insect_c.loc;
					float loc2 = clipped_array[clipped_cnt - 1].loc;
					//this epsilon could not be too large because loc varies in a small range within [0, 1]
                    if( loc1 - loc2 > EPS)		
                        clipped_array[clipped_cnt++] = insect_c;
                    else if(loc2 - loc1 > EPS)
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
inline void AddIntersection(trgl ts, trgl tc, pt *clipped_array, int &clipped_cnt)
{
    for(int ic = 0; ic < 3; ic++)
    {
        for(int is = 0; is < 3; is++)
        {
            pt insect_c;
            //Intersect(tc.p[ic], tc.p[(ic+1)%3], ts.p[is], ts.p[(is+1)%3 ],
            //        insect_c, insect_s);
			IntersectIncludeBoundary(tc.p[ic].coord, tc.p[(ic+1)%3].coord, ts.p[is].coord, ts.p[(is+1)%3 ].coord,
                    insect_c);

            if(insect_c.loc >= 0)
            {
                insect_c.loc += ic;
                if(clipped_cnt > 0)
                {
					float loc1 = insect_c.loc;
					float loc2 = clipped_array[clipped_cnt - 1].loc;
					//this epsilon could not be too large because loc varies in a small range within [0, 1]
                    if( loc1 - loc2 > EPS)		
                        clipped_array[clipped_cnt++] = insect_c;
                    else if(loc2 - loc1 > EPS)
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
template <typename T>
#if NVCC_ON
__host__ __device__
#endif
inline void myswap(T &a, T &b)
{
	T tmp = a;
	a = b;
	b = tmp;
}

#if NVCC_ON
__host__
#endif
inline void printTrgl(trgl t)
{
	cout<<"("<<t.p[0].coord.x << ","<< t.p[1].coord.x << "," << t.p[2].coord.x << "," << t.p[0].coord.x<<endl;
	cout<<"("<<t.p[0].coord.y << ","<< t.p[1].coord.y << "," << t.p[2].coord.y << "," << t.p[0].coord.y<<endl;
}

inline void printTrgl(triangle t)
{
	cout<<" = ["<<t.p[0].x << ","<< t.p[1].x << "," << t.p[2].x << "," << t.p[0].x<<"];"<<endl;
	cout<<" = ["<<t.p[0].y << ","<< t.p[1].y << "," << t.p[2].y << "," << t.p[0].y<<"];"<<endl;
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

__host__ __device__
inline void Geo2Cart(trgl3 &cart, triangle &geo)
{
	float lat;
	float lon;
	for(int i = 0; i < 3; i++)
	{
		lon = geo.p[i].x * M_PI_180;
		lat = geo.p[i].y * M_PI_180;
		cart.p[i].coord.x = RADIUS * cos(lat) * cos(lon);
		cart.p[i].coord.y = RADIUS * cos(lat) * sin(lon);
		cart.p[i].coord.z = RADIUS * sin(lat);
	}
}

__host__ __device__
inline float3 Geo2Cart(float2 geo)
{
	float3 cart;
	float lon = geo.x * M_PI_180;
	float lat = geo.y * M_PI_180;
	cart.x = RADIUS * cos(lat) * cos(lon);
	cart.y = RADIUS * cos(lat) * sin(lon);
	cart.z = RADIUS * sin(lat);
	return cart;
}

__host__ __device__
inline float3 GeoRadian2Cart(float2 geo)
{
	return make_float3(RADIUS * cos(geo.y) * cos(geo.x), RADIUS * cos(geo.y) * sin(geo.x), RADIUS * sin(geo.y));
}

__host__ __device__
inline void Cart2Geo(float2 &geo, float3 &cart)
{
	geo.x = atan2(cart.y, cart.x) * M_180_PI;
	geo.y = asin(cart.z / RADIUS) * M_180_PI;
	if(geo.x < 0)
		geo.x = geo.x + 360;
}

__host__ __device__
inline float3 Cart2Geo(float2 geo)
{
	float3 cart;
	geo.x = atan2(cart.y, cart.x) * M_180_PI;
	geo.y = asin(cart.z / RADIUS) * M_180_PI;
	if(geo.x < 0)
		geo.x = geo.x + 360;
	return cart;
}

__host__ __device__
inline void shrink(pt3 *arr, int size)
{
	for(int i = 1; i < size; i++)
		if(arr[i - 1].loc >= (arr[i].loc ))
			arr[i].loc = -1;
	if(arr[size - 1].loc == (arr[0].loc + 3))
		arr[size - 1].loc = -1;

	int cnt = 1;
	for(int i = 1; i < size; i++)
	{
		if(arr[i].loc != -1)
		{
			arr[cnt++] = arr[i];
		}
	}
}

#if NVCC_ON
__host__ __device__
#endif
void clip3(triangle *t_s1, triangle *t_c1, pt clipped_array_out[6], int &clipped_cnt, instructSet *stateInstr)
{
	trgl3 ts, tc;
	pt3 clipped_array[7];
	Geo2Cart(ts, *t_s1);
	Geo2Cart(tc, *t_c1);
	//mark inside or outside for the triangle vertices
	//and count the number of inside vertices
	int cnt_in_s = 0, cnt_in_c = 0;
	for(int i = 0; i < 3; i++)
	{
		if(tc.p[i].loc = testInside(tc.p[i].coord, ts))
		   cnt_in_c++;

		if(ts.p[i].loc = testInside(ts.p[i].coord, tc))
			cnt_in_s++;
	}

	//make the "in" vertices in the front of the array
	int a[3] = {0, 1, 0};
	for(int i = 0; i < 3; i++)
	{
		int idx = a[i];
		if(tc.p[idx].loc == 0 && tc.p[idx + 1].loc == 1)
			myswap<pt3>(tc.p[idx], tc.p[idx + 1]);
		if(ts.p[idx].loc == 0 && ts.p[idx + 1].loc == 1)
			myswap<pt3>(ts.p[idx], ts.p[idx + 1]);
	}

	bool test;
	if(1 == cnt_in_c && 1 == cnt_in_s)
		//test = BIntersectIncludeBoundary(ts.p[1], ts.p[2], tc.p[0], tc.p[1]);
		test = BIntersect(ts.p[1].coord, ts.p[2].coord, tc.p[0].coord, tc.p[1].coord);

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
//		if(clipped_array[clipped_cnt - 1].loc < ts.p[0].loc)
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

//	shrink(clipped_array, clipped_cnt);

	//if number of edge less than 3, then this is not a polygon
	if(clipped_cnt > 0 && clipped_cnt < 3)
	{

	
		//cout<<"state:"<<state<<endl;
		//cout<<"clipped_cnt:"<<clipped_cnt<<endl;
		//cout<<"error:polygon has one or two vertices, impossible case!"<<endl;
		clipped_cnt = 0;
	//	exit(1);
	}
	else if(clipped_cnt > 6)
	{
		//printTrgl(*t_s1);
		//printTrgl(*t_c1);
		//cout<<"state:"<<state<<endl;
		//cout<<"clipped_cnt:"<<clipped_cnt<<endl;
		//exit(1);
		clipped_cnt = 6;
	}

	for(int i = 0; i < clipped_cnt; i++)
	{
		Cart2Geo(clipped_array_out[i].coord, clipped_array[i].coord);
	}
	//clipped_cnt = ts.p[0].x * 1000;//testInside(ts.p[0], tc);
//	clipped_array[0] = ts.p[0];
}


#if NVCC_ON
__host__ __device__
#endif
void clip(triangle *t_s1, triangle *t_c1, pt clipped_array[6], int &clipped_cnt, instructSet *stateInstr)
{
	trgl ts, tc;
    for(int i = 0; i < 3; i++)
    {
		ts.p[i].coord.x = t_s1->p[i].x;
        ts.p[i].coord.y = t_s1->p[i].y;
        tc.p[i].coord.x = t_c1->p[i].x;
        tc.p[i].coord.y = t_c1->p[i].y;
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
		if(tc.p[idx].loc == 0 && tc.p[idx + 1].loc == 1)
			myswap(tc.p[idx], tc.p[idx + 1]);
		if(ts.p[idx].loc == 0 && ts.p[idx + 1].loc == 1)
			myswap(ts.p[idx], ts.p[idx + 1]);
	}

	bool test;
	if(1 == cnt_in_c && 1 == cnt_in_s)
		//test = BIntersectIncludeBoundary(ts.p[1], ts.p[2], tc.p[0], tc.p[1]);
		test = BIntersect(ts.p[1].coord, ts.p[2].coord, tc.p[0].coord, tc.p[1].coord);

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

	//triangle *t_s1 = ;
	//triangle *t_c1 = ;



	pt clipped_array[6];
	int clipped_cnt = 0;
	//clip(&t_s[pair[idx].x], &t_c[pair[idx].y], clipped_array, clipped_cnt, d_state);
	clip3(&t_s[pair[idx].x], &t_c[pair[idx].y], clipped_array, clipped_cnt, d_state);
	//if(clipped_cnt > 6)
	//{
	//	clipped_cnt = 7;
	//}
	//
	for(int i = 0; i < clipped_cnt; i++)
	{
		clipped[idx].p[i].x = clipped_array[i].coord.x;
		clipped[idx].p[i].y = clipped_array[i].coord.y;
	}
	//if(clipped_cnt > 6)
	//	asm("trap;");
	clipped_n[idx] = clipped_cnt;
}


__host__
vector<float2> clip_serial(triangle t_s, triangle t_c)
{
    vector<float2> clipped;
    //trgl ts, tc;
    //for(int i = 0; i < 3; i++)
    //{
    //    ts.p[i].x = t_s.p[i].x;
    //    ts.p[i].y = t_s.p[i].y;
    //    tc.p[i].x = t_c.p[i].x;
    //    tc.p[i].y = t_c.p[i].y;
    //}
	pt clipped_array[6];
	int clipped_cnt = 0;
	//clip(&t_s, &t_c, clipped_array, clipped_cnt, _stateSet);
	clip3(&t_s, &t_c, clipped_array, clipped_cnt, _stateSet);

    for(int i = 0; i < clipped_cnt; i++)
    {
		float2 p = make_float2(clipped_array[i].coord.x, clipped_array[i].coord.y);
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

struct remove_z
{
	__host__ __device__
	float2 operator() (thrust::tuple<double, double, double> p)
	{
		return make_float2(thrust::get<0>(p), thrust::get<1>(p));
	}
};


struct quad_to_triangles
{
	__host__ __device__
	thrust::tuple<int, int, int, int, int, int> operator() 
		(thrust::tuple<vtkIdType, vtkIdType, vtkIdType, vtkIdType, vtkIdType> cellIdx,
		 thrust::tuple<float, float> coords)
	{
		int i0 = thrust::get<1>(cellIdx);
		int i1 = thrust::get<2>(cellIdx);
		int i2 = thrust::get<3>(cellIdx);
		int i3 = thrust::get<4>(cellIdx);
		return thrust::make_tuple<int, int, int, int, int, int>(i0,i1,i2,i0,i2,i3);
	}
};



typedef thrust::tuple<vtkIdType, vtkIdType, vtkIdType, vtkIdType, vtkIdType> vec5_idtype;
typedef thrust::tuple<float2, float2, float2, float2, float2, float2> trgl2;//coordinates of two triangles
typedef thrust::tuple<float2, float2, float2> trgl1;//coordinates of two triangles

struct assign_triangle_coords
{
	float2* coords;
	assign_triangle_coords(float2* _coords)
	{
		coords = _coords;
	}

	template <typename Tuple>
	__device__ void operator()(Tuple t)
	{
		vec5_idtype cellIdx = thrust::get<0>(t);
		int i0 = thrust::get<1>(cellIdx);
		int i1 = thrust::get<2>(cellIdx);
		int i2 = thrust::get<3>(cellIdx);
		int i3 = thrust::get<4>(cellIdx);

		//compare the distance of diagonal vertices
		//pick the shorter one to devide the quad
		//use radius, instead of degree
		float2 p0 = coords[i0] * M_PI_180;
		float2 p1 = coords[i1] * M_PI_180;
		float2 p2 = coords[i2] * M_PI_180;
		float2 p3 = coords[i3] * M_PI_180;

		float3 p0_cart = GeoRadian2Cart(p0);
		float3 p1_cart = GeoRadian2Cart(p1);
		float3 p2_cart = GeoRadian2Cart(p2);
		float3 p3_cart = GeoRadian2Cart(p3);

		float3 p0_p2 = p2_cart - p0_cart;
		float3 p1_p3 = p3_cart - p1_cart;

		trgl2 twoTrgls;
		if(dot(p0_p2, p0_p2) < dot(p1_p3, p1_p3))
		{
			get<0>(twoTrgls) = p0;
			get<1>(twoTrgls) = p1;
			get<2>(twoTrgls) = p2;
			get<3>(twoTrgls) = p0;
			get<4>(twoTrgls) = p2;
			get<5>(twoTrgls) = p3;
		}
		else
		{
			get<0>(twoTrgls) = p0;
			get<1>(twoTrgls) = p1;
			get<2>(twoTrgls) = p3;
			get<3>(twoTrgls) = p1;
			get<4>(twoTrgls) = p2;
			get<5>(twoTrgls) = p3;
		}
		thrust::get<1>(t) = twoTrgls;
	}
	
};


__device__ inline bool Side0(float radianAngle)
{
	if(radianAngle >= (M_PI_4 * 7) || radianAngle < M_PI_4)
		return true;
	return false;
}

__device__ inline bool Side1(float radianAngle)
{
	if(radianAngle >= (M_PI_4) && radianAngle < (M_PI_4 * 3))
		return true;
	return false;
}

__device__ inline bool Side2(float radianAngle)
{
	if(radianAngle >= (M_PI_4 * 3) && radianAngle < (M_PI_4 * 5))
		return true;
	return false;
}

__device__ inline bool Side3(float radianAngle)
{
	if(radianAngle >= (M_PI_4 * 5) && radianAngle < (M_PI_4 * 7))
		return true;
	return false;
}


__device__ int GetFace(float3 axisAngle)
{
	float2 localCoords;
	if(Side1(axisAngle.y) && Side0(axisAngle.z))
		return 0;
	if(Side3(axisAngle.y) && Side2(axisAngle.z))
		return 1;
	if(Side1(axisAngle.z) && Side0(axisAngle.x))
		return 2;
	if(Side3(axisAngle.z) && Side2(axisAngle.x))
		return 3;
	if(Side1(axisAngle.x) && Side0(axisAngle.y))
		return 4;
	if(Side3(axisAngle.x) && Side2(axisAngle.y))
		return 5;
}


__device__ float2 GetLocalCoords(float3 axisAngle, int face)
{
	float2 localCoords;
	switch(face)
	{
	case 0://Side1(axisAngle.y) && Side0(axisAngle.z)
		localCoords = make_float2(axisAngle.y - M_PI_4, axisAngle.z + M_PI_4);
		break;
	case 1://Side3(axisAngle.y) && Side2(axisAngle.z)
		localCoords = make_float2(axisAngle.y - 5 * M_PI_4, axisAngle.z - 3 * M_PI_4);
		break;
	case 2://Side1(axisAngle.z) && Side0(axisAngle.x)
		localCoords = make_float2(axisAngle.z - M_PI_4, axisAngle.x + M_PI_4);
		break;
	case 3://Side3(axisAngle.z) && Side2(axisAngle.x)
		localCoords = make_float2(axisAngle.z - 5 * M_PI_4, axisAngle.x - 3 * M_PI_4);
		break;
	case 4://Side1(axisAngle.x) && Side0(axisAngle.y)
		localCoords = make_float2(axisAngle.x - M_PI_4, axisAngle.y + M_PI_4);
		break;
	case 5://Side3(axisAngle.x) && Side2(axisAngle.y)
		localCoords = make_float2(axisAngle.x - 5 * M_PI_4, axisAngle.y - 3 * M_PI_4);
		break;
	}
	return localCoords;
}

__device__ int2 GetLocalBin(float2 localCoords)
{
	int2 bin;
	int nBinX = ceil((float)M_PI_2 / BIN_STEP_X);
	int nBinY = ceil((float)M_PI_2 / BIN_STEP_Y);

	if(localCoords.x < 0)
		bin.x = 0;
	else if(localCoords.x > M_PI_2)
		bin.x = nBinX - 1;
	else
		bin.x = localCoords.x / BIN_STEP_X;

	if(localCoords.y < 0)
		bin.y = 0;
	else if(localCoords.y > M_PI_2)
		bin.y = nBinY - 1;
	else
		bin.y = localCoords.y / BIN_STEP_Y;

	return bin;
}

__device__ 

struct functor_getAxisAngle
{
	__host__ __device__
	float3 operator() (float2 p)
	{
		//the angle is in [0, 2*PI]
		float3 axisAngle;
		axisAngle.z = p.x;
		
		float3 cart = GeoRadian2Cart(p);

		axisAngle.x = atan2(cart.z, cart.y);
		if(axisAngle.x < 0)
			axisAngle.x += (2 * M_PI);

		axisAngle.y = atan2(cart.x, cart.z);
		if(axisAngle.y < 0)
			axisAngle.y += (2 * M_PI);

		//axisAngle.z = atan2(cart.y, cart.x);
		//if(axisAngle.z < 0)
		//	axisAngle.z += (2 * M_PI);

		return axisAngle;
	}
};

typedef thrust::tuple<float3, float3, float3> TrglAxisAngle;

__device__ int GetNumBin(TrglAxisAngle t, int face)
{
	int2 bin0 = GetLocalBin(GetLocalCoords(thrust::get<0>(t), face));
	int2 bin1 = GetLocalBin(GetLocalCoords(thrust::get<1>(t), face));
	int2 bin2 = GetLocalBin(GetLocalCoords(thrust::get<2>(t), face));

	int2 min;
	int2 max;

	min.x = min3(bin0.x, bin1.x, bin2.x);
	min.y = min3(bin0.y, bin1.y, bin2.y);
	max.x = max3(bin0.x, bin1.x, bin2.x);
	max.y = max3(bin0.y, bin1.y, bin2.y);

	return (max.x - min.x + 1) * (max.y - min.y + 1);
}

__device__ int getBin(int face, int ix, int iy)
{
	int nBinX = ceil((float)M_PI_2 / BIN_STEP_X);
	int nBinY = ceil((float)M_PI_2 / BIN_STEP_Y);
	return (nBinX * nBinY * face + nBinX * iy + ix);
}

__device__ void GetSearchPair(TrglAxisAngle t, int face, int2* &writeCursor, int trglIdx)
{
	int2 bin0 = GetLocalBin(GetLocalCoords(thrust::get<0>(t), face));
	int2 bin1 = GetLocalBin(GetLocalCoords(thrust::get<1>(t), face));
	int2 bin2 = GetLocalBin(GetLocalCoords(thrust::get<2>(t), face));

	int2 min;
	int2 max;

	min.x = min3(bin0.x, bin1.x, bin2.x);
	min.y = min3(bin0.y, bin1.y, bin2.y);
	max.x = max3(bin0.x, bin1.x, bin2.x);
	max.y = max3(bin0.y, bin1.y, bin2.y);

	for(int iy = min.y; iy <= max.y; iy++)
	{
		for(int ix = min.x; ix <= max.x; ix++)
		{
			//int2(bin index, triangle index)
			*writeCursor = make_int2(getBin(face, ix, iy), trglIdx);//make_int2(ix,iy);//
			writeCursor++;
		}
	}
}

struct functor_getNumBin
{
	 __device__
	int operator() (TrglAxisAngle t)
	{
		int nBin;
		float3 v0 = thrust::get<0>(t);
		float3 v1 = thrust::get<1>(t);
		float3 v2 = thrust::get<2>(t);
		int f0 = GetFace(v0);
		int f1 = GetFace(v1);
		int f2 = GetFace(v2);
	//	return f2;// v0.x * 1000;//(abs(f0 - f1) < EPS);//((f0 != f1));//

		if((abs(f0 - f1) < EPS) && (abs(f1 - f2) < EPS)) //three vertices are all in one face
		{
			nBin = GetNumBin(t, f0);
		}
		else if(abs(f0 - f1) < EPS)	//on two face
		{
			nBin = GetNumBin(t, f1) + GetNumBin(t, f2);
		}
		else if(abs(f1 - f2) < EPS)	//on two face
		{
			nBin = GetNumBin(t, f2) + GetNumBin(t, f0);
		}
		else if(abs(f2 - f0) < EPS)	//on two face
		{
			nBin = GetNumBin(t, f0) + GetNumBin(t, f1);
		}
		else	//on three different faces
		{
			nBin = GetNumBin(t, f0) + GetNumBin(t, f1) + GetNumBin(t, f2);
		}
		return nBin;
	}
};

struct functor_fillSearchStruct
{
	int2* searchStruct;
	functor_fillSearchStruct(int2* _searchStruct)
	{
		searchStruct = _searchStruct;
	}

	__device__ void operator() (thrust::tuple<TrglAxisAngle, int, int> tup)
	{
		TrglAxisAngle t = thrust::get<0>(tup);
		int offset = thrust::get<1>(tup);
		int trglIdx = thrust::get<2>(tup);
		int2* writeCursor = searchStruct + offset;

		float3 v0 = thrust::get<0>(t);
		float3 v1 = thrust::get<1>(t);
		float3 v2 = thrust::get<2>(t);
		int f0 = GetFace(v0);
		int f1 = GetFace(v1);
		int f2 = GetFace(v2);

		if((abs(f0 - f1) < EPS) && (abs(f1 - f2) < EPS)) //three vertices are all in one face
		{
			GetSearchPair(t, f0, writeCursor, trglIdx);
		}
		else if(abs(f0 - f1) < EPS)	//on two face
		{
			GetSearchPair(t, f1, writeCursor, trglIdx);
			GetSearchPair(t, f2, writeCursor, trglIdx);
		}
		else if(abs(f1 - f2) < EPS)	//on two face
		{
			GetSearchPair(t, f2, writeCursor, trglIdx);
			GetSearchPair(t, f0, writeCursor, trglIdx);
		}
		else if(abs(f2 - f0) < EPS)	//on two face
		{
			GetSearchPair(t, f0, writeCursor, trglIdx);
			GetSearchPair(t, f1, writeCursor, trglIdx);
		}
		else	//on three different faces
		{
			GetSearchPair(t, f0, writeCursor, trglIdx);
			GetSearchPair(t, f1, writeCursor, trglIdx);
			GetSearchPair(t, f2, writeCursor, trglIdx);
		}
	}
};

struct BinCmp {
	__host__ __device__
	bool operator()(const int2& v1, const int2& v2) {
		return v1.x < v2.x;
	}
};

void GetPairs(vtkPoints* vtkPts_s, vtkCellArray* vtkCls_s, thrust::device_vector<int2> &d_vec_searchStruct)//, int &numBins)
{
	thrust::tuple<double, double, double>* pointCoords_s = (thrust::tuple<double, double, double>*)vtkPts_s->GetVoidPointer(0);
	int nPoints = vtkPts_s->GetNumberOfPoints();
	clock_t t_1 = clock();
	thrust::device_vector<thrust::tuple<double, double, double>> d_vec_vtkPtsCoords_s
		(pointCoords_s, pointCoords_s + nPoints);
	clock_t t0 = clock();
	unsigned long compute_time = (t0 - t_1) * 1000 / CLOCKS_PER_SEC;
    cout<<"loading VTK point data:"<< (float)compute_time * 0.001 << "sec" << endl;

	thrust::device_vector<float2> d_vec_vtkPts_s(nPoints);
	
	
	
	transform(d_vec_vtkPtsCoords_s.begin(), d_vec_vtkPtsCoords_s.end(), d_vec_vtkPts_s.begin(), remove_z());
	clock_t t1 = clock();
	compute_time = (t1 - t0) * 1000 / CLOCKS_PER_SEC;
    cout<<"remove z coordinate:"<< (float)compute_time * 0.001 << "sec" << endl;
	/**********Cells********/
	int nCells_s = vtkCls_s->GetNumberOfCells();

	vtkIdType* cellIdx_s = vtkCls_s->GetData()->GetPointer(0);
	int sizeCellArray_s = vtkCls_s->GetSize();

	//input: point index of quad
	thrust::device_vector<vtkIdType> d_vec_vtkCls_s(cellIdx_s,
		cellIdx_s + sizeCellArray_s);
	//device pointer to one quad indices
	thrust::device_ptr<vec5_idtype> d_vec_vtkCls_vec5_s = 
		thrust::device_ptr<vec5_idtype>((vec5_idtype*)raw_pointer_cast( &d_vec_vtkCls_s[0]));

	//output: point index of two triangles
	thrust::device_vector<trgl2> trglCoords_s(nCells_s);

	//input: points coordinates(globally access)
	float2* ptsCoords_s = thrust::raw_pointer_cast(d_vec_vtkPts_s.data());


	//computing
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_vec_vtkCls_vec5_s, trglCoords_s.begin())), 
		thrust::make_zip_iterator(thrust::make_tuple(d_vec_vtkCls_vec5_s + nCells_s, trglCoords_s.end())), 
		assign_triangle_coords(ptsCoords_s));

	clock_t t2 = clock();
    compute_time = (t2 - t1) * 1000 / CLOCKS_PER_SEC;
    cout<<"time to generate array to put triangle coordinates:"<< (float)compute_time * 0.001 << "sec" << endl;

	//for(int i = 0; i < 4; i++)
	//{
	//	trgl2 t2 = trglCoords_s[i];
	//	cout<< thrust::get<0>(t2).x<<","<<thrust::get<0>(t2).y<<endl;
	//	cout<< thrust::get<1>(t2).x<<","<<thrust::get<1>(t2).y<<endl;
	//	cout<< thrust::get<2>(t2).x<<","<<thrust::get<2>(t2).y<<endl;
	//	cout<< thrust::get<3>(t2).x<<","<<thrust::get<3>(t2).y<<endl;
	//	cout<< thrust::get<4>(t2).x<<","<<thrust::get<4>(t2).y<<endl;
	//	cout<< thrust::get<5>(t2).x<<","<<thrust::get<5>(t2).y<<endl;
	//}

	//input
	thrust::device_ptr<float2> d_ptr_pointGeoCoords_s = 
		thrust::device_ptr<float2>((float2*)raw_pointer_cast(&trglCoords_s[0]));
	//output:
	//each cell has 2 triangle, each triangle has 3 points
	int nVertex = nCells_s * 3 * 2;
	thrust::device_vector<float3> d_vec_pointAxisAngle_s(nVertex);

	//computing
	thrust::transform(d_ptr_pointGeoCoords_s, d_ptr_pointGeoCoords_s + nVertex, 
		d_vec_pointAxisAngle_s.begin(), functor_getAxisAngle());
	clock_t t3 = clock();
    compute_time = (t3 - t2) * 1000 / CLOCKS_PER_SEC;
     cout<<"time to compute axis angle:"<< (float)compute_time * 0.001 << "sec" << endl;
	//cout<<"axis angle:"<<endl;
	//for(int i = 0; i < 4; i++)
	//{
	//	trgl2 t2 = trglCoords_s[i];
	//	cout<< ((float3)d_vec_pointAxisAngle_s.data()[i]).x<<","
	//		<<((float3)d_vec_pointAxisAngle_s.data()[i]).y<<","
	//		<<((float3)d_vec_pointAxisAngle_s.data()[i]).z
	//		<<endl;
	//}
	
	//output: number of Bins for each triangle
	int nTrgl = nCells_s * 2;
	thrust::device_vector<int> d_vec_numBinPerTrgl(nTrgl);
	//input:
	thrust::device_ptr<TrglAxisAngle> d_ptr_trglAxisAngle_s
		((TrglAxisAngle*)raw_pointer_cast(d_vec_pointAxisAngle_s.data())) ;

	//compute the number of bins, each triangle falls in
	thrust::transform(d_ptr_trglAxisAngle_s, d_ptr_trglAxisAngle_s + nTrgl, 
		d_vec_numBinPerTrgl.begin(), functor_getNumBin());

	clock_t t4 = clock();
    compute_time = (t4 - t3) * 1000 / CLOCKS_PER_SEC;
     cout<<"time to compute the number of bins, each triangle falls in:"<< (float)compute_time * 0.001 << "sec" << endl;
	//cout<<"nTrgl:"<<nTrgl<<endl;
	/*cout<<"number of bins:"<<endl;
	for(int i = 2708; i < 2712; i ++)
	{
		cout<<d_vec_numBinPerTrgl[i]<<endl;
	}*/
	//input:
	thrust::device_vector<int> d_vec_searchStructOffset(nTrgl);
	//compute:
	thrust::exclusive_scan(thrust::device, d_vec_numBinPerTrgl.begin(), d_vec_numBinPerTrgl.end(), 
		d_vec_searchStructOffset.begin()); 

		clock_t t5 = clock();
    compute_time = (t5 - t4) * 1000 / CLOCKS_PER_SEC;
     cout<<"time to do scan for offset:"<< (float)compute_time * 0.001 << "sec" << endl;

	//cout<<"offset:"<<endl;
	//for(int i = 0; i < 32; i++)
	//	cout<<d_vec_searchStructOffset[i]<<endl;

	int numBins = d_vec_searchStructOffset.back() + d_vec_numBinPerTrgl.back();
	//cout<<"d_vec_searchStructOffset[nTrgl - 1]:"<<d_vec_searchStructOffset[nTrgl - 1]<<endl;
	//cout<<"d_vec_numBinPerTrgl[nTrgl - 1]:"<<d_vec_numBinPerTrgl[nTrgl - 1]<<endl;
	//cout<<"numBins:"<<numBins<<endl;

	//input::triangle index
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + nTrgl;
	//output: search structure int2(bin index, triangle index)
	//thrust::device_vector<int2> d_vec_searchStruct(numBins);
	d_vec_searchStruct.resize(numBins);
	int2* d_raw_ptr_searchStruct = raw_pointer_cast(d_vec_searchStruct.data());
	//compute search structure:
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_ptr_trglAxisAngle_s, d_vec_searchStructOffset.begin(),first)),
		thrust::make_zip_iterator(thrust::make_tuple(d_ptr_trglAxisAngle_s + nTrgl, d_vec_searchStructOffset.end(), last)), 
		functor_fillSearchStruct(d_raw_ptr_searchStruct));

		clock_t t6 = clock();
    compute_time = (t6 - t5) * 1000 / CLOCKS_PER_SEC;
     cout<<"time to input bins for each triangle:"<< (float)compute_time * 0.001 << "sec" << endl;

	thrust::device_ptr<int2> d_ptr_searchStruct(d_raw_ptr_searchStruct);


	//for(int i = 3500; i < numBins; i++)
	//{
	//	int2 temp = d_ptr_searchStruct[i];

	//	cout<<temp.x <<","<<temp.y<<endl;
	//	if(temp.x == 0)
	//		exit(1);
	//}

	

	//sort based on bin number
	thrust::sort(d_ptr_searchStruct, d_ptr_searchStruct + numBins, BinCmp());
	//for(int i = 0; i < 500; i++)
	//{
	//	int2 temp = d_ptr_searchStruct[i];
	//	cout<<temp.x <<","<<temp.y<<endl;
	//}
		clock_t t7 = clock();
	compute_time = (t7 - t6) * 1000 / CLOCKS_PER_SEC;
     cout<<"time to sort based on bin number:"<< (float)compute_time * 0.001 << "sec" << endl;

//	return d_vec_searchStruct;
//	searchStruct = d_ptr_searchStruct;
}


__host__ void runCUDA(/*vtkPoints* vtkPts_s, vtkCellArray* vtkCls_s, vtkPoints* vtkPts_c, vtkCellArray* vtkCls_c,*/
	char* filename_subject, char* filename_constraint)
{
	thrust::device_vector<int2> searchStruct_s, searchStruct_c;
	int numBins_s, numBins_c;
	

    //clock_t t0 = clock();
	vtkSmartPointer<vtkUnstructuredGridReader> reader =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();
	vtkSmartPointer<vtkUnstructuredGridReader> reader_c =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();

	//reading subject file
    reader->SetFileName(filename_subject);
    reader->Update();
    vtkUnstructuredGrid* grid_s = reader->GetOutput();
    vtkPoints* points_s = grid_s->GetPoints();
    vtkCellArray* cell_s = grid_s->GetCells();
    //reader->CloseVTKFile();


	//reading constriant file
	reader_c->SetFileName(filename_constraint);
    reader_c->Update(); // Needed because of GetScalarRange
    vtkUnstructuredGrid* grid_c = reader_c->GetOutput();
    vtkPoints* points_c = grid_c->GetPoints();
    vtkCellArray* cell_c = grid_c->GetCells();
   // reader->CloseVTKFile();
	
	
	GetPairs(points_s, cell_s, searchStruct_s);//, numBins_s);
	GetPairs(points_c, cell_c, searchStruct_c);//, numBins_c);
	
	cout<<"numBins_s:"<<searchStruct_s.size()<<endl;
	cout<<"numBins_c:"<<searchStruct_c.size()<<endl;

	
	/*clock_t t1 = clock();
    unsigned long compute_time = (t1 - t0) * 1000 / CLOCKS_PER_SEC;
    cout<<"time to get pair <bin number, triangle number>:"<< (float)compute_time * 0.001 << "sec" << endl;
*/
	/*cout<<"pair_s:"<<endl;
	for(int i = 0; i < 50; i++)
	{
		int2 temp = searchStruct_s[i];
		cout<<temp.x <<","<<temp.y<<endl;
	}

	cout<<"pair_c:"<<endl;
	for(int i = 0; i < 50; i++)
	{
		int2 temp = searchStruct_c[i];
		cout<<temp.x <<","<<temp.y<<endl;
	}*/
}