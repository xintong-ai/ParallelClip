#include "vtkNew.h"
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkUnstructuredGridWriter.h"
#include <vtkSmartPointer.h>
#include <iostream>
#include "vtkCellArray.h"
#include "time.h"
#include <cmath>
#include <vtkFloatArray.h>

#define PARALLEL_ON 1

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
    for(int i = 0; i < q.size() / 2; i++)
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

void writePolygonFileFastArray(char* filename, float* points_array, vtkIdType* cells_array, int nCells, int nPts)
{
	cout<<"cells_array:"<<endl;
	for(int i = 0; i < 30; i++)
		cout<<cells_array[nCells + nPts - 1 - i]<<endl;
	vtkIdTypeArray *cellIdx = vtkIdTypeArray::New();
	cout<<"**2"<<endl;

	//cells
	cellIdx->SetArray(cells_array, nCells + nPts, 1);
	vtkCellArray *cells = vtkCellArray::New();
	cells->SetCells(nCells, cellIdx);
	cout<<"**3"<<endl;

	//points
	vtkFloatArray *vtk_pts_array = vtkFloatArray::New();
	vtk_pts_array->SetNumberOfComponents(3);
	vtk_pts_array->SetArray(points_array, nPts * 3,1 );
	vtkPoints *pts = vtkPoints::New();
	pts->SetDataTypeToFloat();
	pts->SetData(vtk_pts_array);
	cout<<"**4"<<endl;

	//grid
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(pts);
	grid->SetCells((int)VTK_POLYGON, cells);
	cout<<"**5"<<endl;
	
	//writer
    vtkNew<vtkUnstructuredGridWriter> writer;
    writer->SetFileName(filename);
    writer->SetInputData(grid);
    writer->SetFileTypeToBinary();
    writer->Write();
}

void writePolygonFileFast(char* filename, vector<vector<point> > poly)
{
	int nPts = 0;
	vtkIdTypeArray *cellIdx = vtkIdTypeArray::New();
	vector<vtkIdType> idx_array;
	vector<float> pts_vec;
	for(int i = 0; i < poly.size(); i++)
	{
		int np = poly[i].size();
		idx_array.push_back(np);
		for(int p = 0; p < np; p++, nPts++)
		{
			idx_array.push_back(nPts);
			pts_vec.push_back(poly[i][p].x);
			pts_vec.push_back(poly[i][p].y);
			pts_vec.push_back(0);
		}
	}

	writePolygonFileFastArray(filename, &pts_vec[0], &idx_array[0], poly.size(), nPts);
//	cellIdx->SetArray(&idx_array[0], idx_array.size(), 1);
//	vtkCellArray *cells = vtkCellArray::New();
//	cells->SetCells(poly.size(), cellIdx);
//
////	float* poly_ptr = &poly[0][0].x;
////	float* pts_array = (float*)malloc(nPts * 3 * sizeof(float));
//	/*for(int i = 0; i < nPts; i++)
//	{
//		pts_array[i * 3] = poly_ptr[i * 2];
//		pts_array[i * 3 + 1] = poly_ptr[i * 2 + 1];
//		pts_array[i * 3 + 2] = 0;
//	}*/
//
//	vtkFloatArray *vtk_pts_array = vtkFloatArray::New();
//	vtk_pts_array->SetNumberOfComponents(3);
//	vtk_pts_array->SetArray(&pts_vec[0], pts_vec.size(),1 );
//	
//	vtkPoints *pts = vtkPoints::New();
//	pts->SetDataTypeToFloat();
//	pts->SetData(vtk_pts_array);
//	//pts->SetNumberOfPoints(nPts);
//
////	cells->SetNumberOfCells(poly.size());
//    //pts.Allocate(poly.size());
//    //vector<point> points;
//    //for(int i = 0; points)
//    //for each polygon
//    //int count = 0;
//    //for(int p = 0; p < poly.size(); p++)
//    //{
//    //    vector<point> onePoly = poly[p];
//    //    //for each vertex
//    //    vtkSmartPointer<vtkIdList> idl = vtkSmartPointer<vtkIdList>::New();
//    //    for(int v = 0; v < onePoly.size(); v++)
//    //    {
//    //        point p = onePoly[v];
//    //        pts->InsertNextPoint(p.x,p.y,0.0);
//    //        idl->InsertNextId(count++);
//    //    }
//    //    grid->InsertNextCell((int)VTK_POLYGON, idl);
//    //}
//    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
//    grid->SetPoints(pts);
//	grid->SetCells((int)VTK_POLYGON, cells);
//
//    //grid.SetCells();
//
//    vtkNew<vtkUnstructuredGridWriter> writer;
//    writer->SetFileName(filename);
//    writer->SetInputData(grid);
//    writer->SetFileTypeToBinary();
//    writer->Write();

}

//clip two set of cellsNoSort
void clipSets(vector<triangle> t_s, vector<triangle> t_c, vector<vector<int> > cellInBin_c)
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
				float sx[2], sy[2], cx[2], cy[2];
				triangle ts = t_s[is];
				triangle tc = t_c[ic];
				sx[0] = min3<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
				cx[1] = max3<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
				if(sx[0] >= cx[1])
					continue;

				sy[0] = min3<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
				cy[1] = max3<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
				if(sy[0] >= cy[1])
					continue;

				cx[0] = min3<float>(tc.p[0].x, tc.p[1].x, tc.p[2].x);
				sx[1] = max3<float>(ts.p[0].x, ts.p[1].x, ts.p[2].x);
				if(cx[0] >= sx[1])
					continue;

				cy[0] = min3<float>(tc.p[0].y, tc.p[1].y, tc.p[2].y);
				sy[1] = max3<float>(ts.p[0].y, ts.p[1].y, ts.p[2].y);
				if(cy[0] >= sy[1])
					continue;

                IndexPair pr(is, ic);
                polyPairs.push_back(pr);
            }
        }
		if(is % 100 == 0)	
			cout<<"is = "<< is << endl;
    }

	clock_t t1 = clock();
    
#if PARALLEL_ON
    loadDataToDevice(&t_s[0].p[0].x, &t_c[0].p[0].x, t_s.size(), &polyPairs[0].is, polyPairs.size());
#endif

#if PARALLEL_ON
	float* points;
	vtkIdType* cells;
	int nCells;
	int nPts;
	cout<<"**0"<<endl;
	runKernel(points, cells, nCells, nPts);
	clock_t t2 = clock();
	writePolygonFileFastArray("data/CAM_0_small_clipped_parallel.vtk", points, cells, nCells, nPts);
#else
    for(int i = 0; i < polyPairs.size(); i++)
    {
		vector<point> clipped;
        clipped = clip_serial(t_s[polyPairs[i].is], t_c[polyPairs[i].ic]);
        if(clipped.size()>0)
            clippedAll.push_back(clipped);
        if((i % 100000) == 0)
            cout<<"i = "<<i<<endl;
    }
	clock_t t2 = clock();
	writePolygonFileFast("data/CAM_0_small_clipped.vtk", clippedAll);
#endif
    unsigned long compute_time = (t2 - t1) * 1000 / CLOCKS_PER_SEC;
	cout<<"Clipping time:"<< (float)compute_time * 0.001 << "sec" << endl;
  //  return clippedAll;
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

#if PARALLEL_ON
    initCUDA();
#endif

    clock_t t2 = clock();
    vector<vector<int> > cellsInBin = Binning(trias_c);
    //vector<vector<point> > clippedPoly = 
	clipSets(trias_s, trias_c, cellsInBin);

    clock_t t3 = clock();
    unsigned long compute_time = (t3 - t2) * 1000 / CLOCKS_PER_SEC;
    cout<<"computing time:"<< (float)compute_time * 0.001 << "sec" << endl;

 //   writePolygonFile("data/CAM_0_small_clipped.vtk", clippedPoly);
    return 1;
}
