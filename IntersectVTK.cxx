#include "vtkNew.h"
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkUnstructuredGridWriter.h"
#include <vtkSmartPointer.h>
#include <iostream>
#include "vtkCellArray.h"
#include "time.h"
#include <cmath>

#define PARALLEL_ON 0

#include "clip.h"

using namespace std;


#define STEP_X 1
#define STEP_Y 1
#define X_MAX 360
#define X_MIN 0
#define Y_MAX 90
#define Y_MIN -90



static int rangeX = X_MAX - X_MIN;
static int rangeY = Y_MAX - Y_MIN;
static int nbinX = rangeX / STEP_X;
static int nbinY = rangeY / STEP_Y;
static int nbin = nbinX * nbinY;

extern "C"
void loadDataToDevice(float* trgl_s, float* trgl_c, int ntrgl, int *pair, int npar);

extern "C"
void initCUDA();


//
//struct point
//{
//    float x;
//    float y;
//    point(float _x, float _y)
//    {
//        x = _x;
//        y = _y;
//    }
//    point(){};
//};
//
//
//struct triangle
//{
//    point p[3];
//    triangle(point p0, point p1, point p2)
//    {
//        p[0] = p0;
//        p[1] = p1;
//        p[2] = p2;
//    }
//    triangle(point _p[4])
//    {
//        p[0] = _p[0];
//        p[1] = _p[1];
//        p[2] = _p[2];
//    }
//};
//
//
//
struct IndexPair
{
    int is;
    int ic;
    IndexPair(int _is, int _ic)
    {
        is = _is;
        ic = _ic;
    }
};
//
//
inline int GetXBin(double x)
{
    int ib = (x - X_MIN) / STEP_X;
    ib = ib % nbinX;
    return ib;
}

inline int GetYBin(double y)
{
    int ib = (y - Y_MIN) / STEP_Y;
    ib = ib % nbinY;
    return ib;
}
//
//template<typename T>
//inline T min(T x1, T x2, T x3)
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
//
//template<typename T>
//inline T max(T x1, T x2, T x3)
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
//
vector<int> GetBinTriangle(triangle q)
{
    vector<int> ret;
    int xmin, xmax, ymin, ymax;
    //value can be out of boundary, so mode the number of bins in each coordinate
    xmin = min3<float>(
            GetXBin(q.p[0].x),
            GetXBin(q.p[1].x),
            GetXBin(q.p[2].x));
    xmax = max3<float>(
            GetXBin(q.p[0].x),
            GetXBin(q.p[1].x),
            GetXBin(q.p[2].x));
    ymin = min3<float>(
            GetYBin(q.p[0].y),
            GetYBin(q.p[1].y),
            GetYBin(q.p[2].y));
    ymax = max3<float>(
            GetYBin(q.p[0].y),
            GetYBin(q.p[1].y),
            GetYBin(q.p[2].y));
    xmin = (xmin + nbinX) % nbinX;
    xmax = (xmax + nbinX) % nbinX;
    ymin = (ymin + nbinY) % nbinY;
    ymax = (ymax + nbinY) % nbinY;
    //if the two points are too far away, ignore it
    if((xmax - xmin) > (nbinX / 2) || (ymax - ymin) > (nbinY / 2) )
        return ret;
    for(int y = ymin; y <= ymax; y++)
    {
        for(int x = xmin; x <= xmax; x++)
        {
            ret.push_back(y * nbinX + x);
        }
    }
    return ret;
}
//
vector<vector<int> > Binning(vector<triangle> q)
{
    vector<vector<int> > CellsInBin(nbin, vector<int>(0));
    for(int i = 0; i < q.size(); i++)
    {
        vector<int> bins = GetBinTriangle(q[i]);
        for(int b = 0; b < bins.size(); b++)
        {
            CellsInBin[bins[b]].push_back(i);
        }
    }
    return CellsInBin;
}
//
//import vtk format to the local vector format
void ImportTriangles(vtkPoints* vtkPts, vtkCellArray* vtkCls, vector<triangle> &trias)
{
    vtkNew<vtkIdList> pts;
    point p[4];
    double *coord;
    for(int c = 0; c < vtkCls->GetNumberOfCells(); c++)
    {
        vtkCls->GetCell(c * 5, pts.GetPointer());
        for(int i = 0; i < 4; i++)
        {
            coord = vtkPts->GetPoint(pts->GetId(i));//();
            p[i].x = coord[0];
            p[i].y = coord[1];
        }
        triangle t1(p[0], p[1], p[2]);
        trias.push_back(t1);
        triangle t2(p[0], p[2], p[3]);
        trias.push_back(t2);
    }
}
//
//struct pt
//{
//    float x;
//    float y;
// //   bool in;    //either inside or intersection point
//    float loc;
//    pt(float _x, float _y)
//    {
//        x = _x;
//        y = _y;
//        loc = -1;
//    }
//    pt()
//    {
//        loc = -1;
//    }
//};
//
//struct trgl
//{
//    //the first 3 points are the vertex
//    //others are reserved forintersection points
//    pt p[9];
//};
//


//

//
////line(p1, p2) is parallel with line(q1, q2)
//inline bool parallel(pt p1, pt p2, pt q1, pt q2)
//{
//  float par = (float) ((p2.x - p1.x)*(q2.y - q1.y) -
//                 (p2.y - p1.y)*(q2.x - q1.x));
//  if(abs(par)<EPS)
//      return true;
//  else
//      return false;
//}
//
//struct instructSet
//{
//    bool doIns[14];
//    instructSet()
//    {
//        for(int i = 0; i < 14; i++)
//            doIns[i] = false;
//    }
//};
//static instructSet stateSet[11];
//
#if !PARALLEL_ON
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
    //float sx[2], sy[2], cx[2], cy[2];
    //sx[0] = min<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
    //cx[1] = max<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
    //if(sx[0] >= cx[1])
    //    return clipped;

    //sy[0] = min<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
    //cy[1] = max<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
    //if(sy[0] >= cy[1])
    //    return clipped;

    //cx[0] = min<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
    //sx[1] = max<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
    //if(cx[0] >= sx[1])
    //    return clipped;

    //cy[0] = min<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
    //sy[1] = max<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
    //if(cy[0] >= sy[1])
    //    return clipped;

    //i = 0;
    ////mark inside or outside for the triangle vertices
    ////and count the number of inside vertices
    //int cnt_in_s = 0, cnt_in_c = 0;
    //for(i = 0; i < 3; i++)
    //{
    //    if(tc.p[i].loc = testInside(tc.p[i], ts))
    //       cnt_in_c++;

    //    if(ts.p[i].loc = testInside(ts.p[i], tc))
    //        cnt_in_s++;
    //}

    ////make the "in" vertices in the front of the array
    //int a[3] = {0, 1, 0};
    //for(i = 0; i < 3; i++)
    //{
    //    int idx = a[i];
    //    if(!tc.p[idx].loc && tc.p[idx + 1].loc)
    //        swap(tc.p[idx], tc.p[idx + 1]);
    //    if(!ts.p[idx].loc && ts.p[idx + 1].loc)
    //        swap(ts.p[idx], ts.p[idx + 1]);
    //}

    //bool test;
    //if(1 == cnt_in_c && 1 == cnt_in_s)
    //    test = BIntersectIncludeBoundary(ts.p[1], ts.p[2], tc.p[0], tc.p[1]);

    //int state = -1;
    //if(0 == cnt_in_c && 0 == cnt_in_s)
    //    state = 0;
    //else if(0 == cnt_in_c && 1 == cnt_in_s)
    //    state = 1;
    //else if(1 == cnt_in_c && 0 == cnt_in_s)
    //    state = 2;
    //else if(0 == cnt_in_c && 2 == cnt_in_s)
    //    state = 3;
    //else if(2 == cnt_in_c && 0 == cnt_in_s)
    //    state = 4;
    //else if(0 == cnt_in_c && 3 == cnt_in_s)
    //    state = 5;
    //else if(3 == cnt_in_c && 0 == cnt_in_s)
    //    state = 6;
    //else if(1 == cnt_in_c && 2 == cnt_in_s)
    //    state = 7;
    //else if(2 == cnt_in_c && 1 == cnt_in_s)
    //    state = 8;
    //else if(1 == cnt_in_c && 1 == cnt_in_s && !test)
    //    state = 9;
    //else// if(1 == cnt_in_c && 1 == cnt_in_s && !test1) and (1 == cnt_in_c && 1 == cnt_in_s && test1 && test2)
    //    state = 10;
    ////+cs

    //pt clipped_array[6];

    //int clipped_cnt = 0;
    //instructSet is = stateSet[state];
    //if(is.doIns[0])//+sc
    //    AddIntersection(tc, ts, clipped_array, clipped_cnt);
    //if(is.doIns[1])//+cs
    //    AddIntersection(ts, tc, clipped_array, clipped_cnt);
    //if(is.doIns[12])
    //    clipped_array[clipped_cnt] = clipped_array[clipped_cnt - 1];
    //if(is.doIns[2])//+c0-
    //    clipped_array[clipped_cnt - 1] = tc.p[0];
    //if(is.doIns[3])//+s0-
    //    clipped_array[clipped_cnt - 1] = ts.p[0];
    //if(is.doIns[13])
    //    clipped_cnt++;
    //if(is.doIns[4])//+s0
    //    clipped_array[clipped_cnt++] = ts.p[0];
    //if(is.doIns[5])//+c0
    //    clipped_array[clipped_cnt++] = tc.p[0];
    //if(is.doIns[6])//+s1
    //    clipped_array[clipped_cnt++] = ts.p[1];
    //if(is.doIns[7])//+c1
    //    clipped_array[clipped_cnt++] = tc.p[1];
    //if(is.doIns[8])//+s2
    //    clipped_array[clipped_cnt++] = ts.p[2];
    //if(is.doIns[9])//+c2
    //    clipped_array[clipped_cnt++] = tc.p[2];
    //if(is.doIns[10])//+r0
    //    clipped_array[clipped_cnt++] = clipped_array[0];
    //if(is.doIns[11])//+r0_s0
    //    clipped_array[0] = ts.p[0];
	pt clipped_array[6];
	int clipped_cnt = 0;
	clip(ts, tc, clipped_array, clipped_cnt);

    for(int i = 0; i < clipped_cnt; i++)
    {
        point p(clipped_array[i].x, clipped_array[i].y);
        clipped.push_back(p);
    }
    return clipped;
}
#endif
//
//
//void setStateInstr()
//{
//    stateSet[0].doIns[1] = true;
//
//    stateSet[1].doIns[0] = true;
//    stateSet[1].doIns[4] = true;
//
//    stateSet[2].doIns[1] = true;
//    stateSet[2].doIns[5] = true;
//
//    stateSet[3].doIns[0] = true;
//    stateSet[3].doIns[4] = true;
//    stateSet[3].doIns[6] = true;
//
//    stateSet[4].doIns[1] = true;
//    stateSet[4].doIns[5] = true;
//    stateSet[4].doIns[7] = true;
//
//    stateSet[5].doIns[4] = true;
//    stateSet[5].doIns[6] = true;
//    stateSet[5].doIns[8] = true;
//
//    stateSet[6].doIns[5] = true;
//    stateSet[6].doIns[7] = true;
//    stateSet[6].doIns[9] = true;
//
//    stateSet[7].doIns[0] = true;
//    stateSet[7].doIns[12] = true;
//    stateSet[7].doIns[2] = true;
//    stateSet[7].doIns[13] = true;
//    stateSet[7].doIns[4] = true;
//    stateSet[7].doIns[6] = true;
//
//    stateSet[8].doIns[1] = true;
//    stateSet[8].doIns[12] = true;
//    stateSet[8].doIns[3] = true;
//    stateSet[8].doIns[13] = true;
//    stateSet[8].doIns[5] = true;
//    stateSet[8].doIns[7] = true;
//
//    stateSet[9].doIns[1] = true;
//    stateSet[9].doIns[5] = true;
//    stateSet[9].doIns[10] = true;
//    stateSet[9].doIns[11] = true;
//
//    stateSet[10].doIns[1] = true;
//    stateSet[10].doIns[3] = true;
//    stateSet[10].doIns[5] = true;
//}

//clip two set of cellsNoSort
vector<vector<point> > clipSets(vector<triangle> t_s, vector<triangle> t_c, vector<vector<int> > cellInBin_c)
{

    setStateInstr();

    vector<vector<point> > clippedAll;
    int ic;
    //for each quad in subject set11

    vector<IndexPair> polyPairs;     //pair of polygons that need to be tested
    for(int is = 0; is < t_s.size(); is++)
    {
        triangle s = t_s[is];// one quad in subject set
        vector<int> bin_s =  GetBinTriangle(s); //the bins the subject quad belong to

        //for each intersected bin
        for(int ibs = 0; ibs < bin_s.size(); ibs++)
        {
            int b_s = bin_s[ibs];//bin number
            vector<int> cellIdx_c = cellInBin_c[b_s];//indices of all the constaint quads in this bin
            for(int i = 0; i < cellIdx_c.size(); i++)
            {
                ic = cellIdx_c[i];
                IndexPair pr(is, ic);
                polyPairs.push_back(pr);
            }
        }
		if(is % 100 == 0)	
			cout<<"is = "<< is << endl;
    }

#if PARALLEL_ON
    loadDataToDevice(&t_s[0].p[0].x, &t_c[0].p[0].x, t_s.size(), &polyPairs[0].is, polyPairs.size());
#endif

#if PARALLEL_ON
	
#else
    for(int i = 0; i < polyPairs.size(); i++)
    {
		vector<point> clipped;
        clipped = clip_serial(t_c[polyPairs[i].ic], t_s[polyPairs[i].is]);
        if(clipped.size()>0)
            clippedAll.push_back(clipped);
        if((i % 100000) == 0)
            cout<<"i = "<<i<<endl;
    }
#endif
    return clippedAll;

}


void writePolygonFile(char* filename, vector<vector<point> > poly)
{
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
    vtkNew<vtkCellArray> cells;
    //pts.Allocate(poly.size());
    //vector<point> points;
    //for(int i = 0; points)
    //for each polygon
    int count = 0;
    for(int p = 0; p < poly.size(); p++)
    {
        vector<point> onePoly = poly[p];
        //for each vertex
        vtkSmartPointer<vtkIdList> idl = vtkSmartPointer<vtkIdList>::New();
        for(int v = 0; v < onePoly.size(); v++)
        {
            point p = onePoly[v];
            pts->InsertNextPoint(p.x,p.y,0.0);
            idl->InsertNextId(count++);
        }
        grid->InsertNextCell((int)VTK_POLYGON, idl);
    }
    grid->SetPoints(pts);

    //grid.SetCells();

    vtkNew<vtkUnstructuredGridWriter> writer;
    writer->SetFileName(filename);
    writer->SetInputData(grid);
    writer->SetFileTypeToBinary();
    writer->Write();

}

int main( int argc, char *argv[] )
{
//	char filename_constraint[100] = "data/CAM_0_small_vec.vtk";//CAM_1_vec.vtk";//"/home/xtong/data/VTK-LatLon2/CAM_1_vec.vtk";
//    char filename_subject[100] = "data/CAM_0_small_vec_warped_5times.vtk";//CAM_1_vec_warped_5times.vtk";//"/home/xtong/data/VTK-LatLon2/CAM_1_vec_warped_5times.vtk";

    char filename_constraint[100] = "data/CAM_1_vec.vtk";//"/home/xtong/data/VTK-LatLon2/CAM_1_vec.vtk";
    char filename_subject[100] = "data/CAM_1_vec_warped_5times.vtk";//"/home/xtong/data/VTK-LatLon2/CAM_1_vec_warped_5times.vtk";

    vector<triangle> trias_c;
    vector<triangle> trias_s;
    vtkSmartPointer<vtkUnstructuredGridReader> reader =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();

    /************constraint polygon****************/
    reader->SetFileName(filename_constraint);
    reader->Update(); // Needed because of GetScalarRange
    vtkUnstructuredGrid* grid_c = reader->GetOutput();
    vtkPoints* points_c = grid_c->GetPoints();
    vtkCellArray* cell_c = grid_c->GetCells();
    cout<<"nc="<<cell_c->GetNumberOfCells()<<endl;
    reader->CloseVTKFile();

    ImportTriangles(points_c, cell_c, trias_c);
    /************subject polygon****************/

    reader->SetFileName(filename_subject);
    reader->Update();
    vtkUnstructuredGrid* grid_s = reader->GetOutput();
    vtkPoints* points_s = grid_s->GetPoints();
    vtkCellArray* cell_s = grid_s->GetCells();
    clock_t t1 = clock();


    ImportTriangles(points_s, cell_s, trias_s);

    clock_t t2 = clock();
#if PARALLEL_ON
    initCUDA();
#endif

    vector<vector<int> > cellsInBin = Binning(trias_c);
    vector<vector<point> > clippedPoly = clipSets(trias_s, trias_c, cellsInBin);

    clock_t t3 = clock();
    unsigned long compute_time = (t3 - t2) * 1000 / CLOCKS_PER_SEC;
    cout<<"computing time:"<< (float)compute_time * 0.001 << "sec" << endl;

    writePolygonFile("data/CAM_0_small_clipped.vtk", clippedPoly);
    return 1;
}

