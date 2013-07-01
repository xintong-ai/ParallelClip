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


#if NVCC_ON 
__constant__ instructSet STATE_SET[11];
#endif

template<typename T>
#if NVCC_ON
    __host__ __device__
#endif
inline void swap(T a, T b)
{
	T tmp = a;
	a = b;
    b = tmp;
}


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
    pt p[9];
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


template<typename T>
#if NVCC_ON
__host__ __device__
#endif
inline T min(T x1, T x2, T x3)
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
inline T max(T x1, T x2, T x3)
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
inline bool testInside(pt p, trgl t)
{
    bool inside = false;
    pt left( -999, p.y);//create(0, point->y, 0, 0, 0, 0, 0, 0, 0, 0.);
    for(int i = 0; i < 3; i++)
    {
        if(BIntersect(left, p, t.p[i], t.p[(i+1)%3]))
            inside = !inside;
    }
    return inside;
}



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
inline void clip(trgl ts, trgl tc, 	pt clipped_array[6], int &clipped_cnt)
{
	float sx[2], sy[2], cx[2], cy[2];
    sx[0] = min<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
	cx[1] = max<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
	if(sx[0] >= cx[1])
	{
		return;
	}

	sy[0] = min<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
	cy[1] = max<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
	if(sy[0] >= cy[1])
	{
		return;
	}

	cx[0] = min<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
	sx[1] = max<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
	if(cx[0] >= sx[1])
	{
		return;
	}

	cy[0] = min<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
	sy[1] = max<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
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
			swap(tc.p[idx], tc.p[idx + 1]);
		if(!ts.p[idx].loc && ts.p[idx + 1].loc)
			swap(ts.p[idx], ts.p[idx + 1]);
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
}

#endif //CLIP_H