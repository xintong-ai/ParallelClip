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
#include "string"

#define PARALLEL_ON 1
#define FILE_DIR "/media/User/Dropbox/sandia/ParallelClip-build/data/"
//#define FILE_DIR "D:/Dropbox/sandia/ParallelClip-build/data/"
 
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

int _nBlock;

inline float dist_2(float2 v1, float2 v2)
{
	float x = v1.x - v2.x;
	float y = v1.y - v2.y;
	return (x * x + y * y);
}

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
	//temperarily used to remove the point out of bundary
	if(min3<float>(q.p[0].x , q.p[1].x, q.p[2].x) < X_MIN)
		return ret;
	if(max3<float>(q.p[0].x , q.p[1].x, q.p[2].x) > X_MAX)
		return ret;
	if(min3<float>(q.p[0].y , q.p[1].y, q.p[2].y) < Y_MIN)
		return ret;
	if(max3<float>(q.p[0].y , q.p[1].y, q.p[2].y) > Y_MAX)
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
    float2 p[4];
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
		if(dist_2(p[0], p[2]) < dist_2(p[1], p[3]) )
		{
			triangle t1(p[0], p[1], p[2]);
			trias.push_back(t1);
			triangle t2(p[0], p[2], p[3]);
			trias.push_back(t2);
		}
		else
		{	
			triangle t1(p[0], p[1], p[3]);
			trias.push_back(t1);
			triangle t2(p[1], p[2], p[3]);
			trias.push_back(t2);
		}
    }
}

void printPolygon(vector<float2> polygon)
{
	cout<<"print polygon:"<<endl;
	for(int i = 0; i < polygon.size(); i++)
	{
		cout<<polygon[i].x<<","<<polygon[i].y<<endl;
	}
}

void writePolygonFile(char* filename, vector<vector<float2> > poly)
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
        vector<float2> onePoly = poly[p];
        //for each vertex
        vtkSmartPointer<vtkIdList> idl = vtkSmartPointer<vtkIdList>::New();
        for(int v = 0; v < onePoly.size(); v++)
        {
            float2 p = onePoly[v];
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
	//	cout<<"nCells:"<<nCells<<endl;
	//cout<<"nPts:"<<nPts<<endl;
	//cout<<"cells_array:"<<endl;
	//for(int i = 0; i < 30; i++)
	//	cout<<cells_array[nCells + nPts - 1 - i]<<endl;

	vtkIdTypeArray *cellIdx = vtkIdTypeArray::New();

	//cells
	cellIdx->SetArray(cells_array, nCells + nPts, 1);
	vtkCellArray *cells = vtkCellArray::New();
	cells->SetCells(nCells, cellIdx);

	//points
	vtkFloatArray *vtk_pts_array = vtkFloatArray::New();
	vtk_pts_array->SetNumberOfComponents(3);
	vtk_pts_array->SetArray(points_array, nPts * 3,1 );
	vtkPoints *pts = vtkPoints::New();
	pts->SetDataTypeToFloat();
	pts->SetData(vtk_pts_array);

	//for(int i = 0; i < 10; i++)
	//	cout<<pts[i]<<endl;
	//grid
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(pts);
	grid->SetCells((int)VTK_POLYGON, cells);
	
	//writer
    vtkNew<vtkUnstructuredGridWriter> writer;
    writer->SetFileName(filename);
    writer->SetInputData(grid);
    writer->SetFileTypeToBinary();
	//writer->SetFileTypeToASCII();
    writer->Write();
}

void writePolygonFileFast(char* filename, vector<vector<float2> > poly)
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
void clipSets(vector<triangle> t_s, vector<triangle> t_c, vector<vector<int> > cellInBin_c, 
	vector<float> &pts_vec,	vector<vtkIdType> &idx_array,	int &nPts)
{

    setStateInstr();

    vector<vector<float2> > clippedAll;
    int ic;
    //for each quad in subject set11

	//printTrgl(t_s[11833]);
	//printTrgl(t_c[12376]);
	//vector<float2> clipped2 = clip_serial(t_s[11833], t_c[12376]);
	//printPolygon(clipped2);
	//exit(2);

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
		if(is % 10000 == 0)	
			cout<<"is = "<< is << endl;
    }
//	cout<<"polyPair.size()"<<polyPairs.size()<<endl;
	//polyPairs.assign(polyPairs.begin() + 17950, polyPairs.end());
//	vector<IndexPair> polyPairs2;
//	polyPairs2.assign(polyPairs.begin(), polyPairs.begin() + 500000);
	clock_t t1 = clock();
    
#if PARALLEL_ON
//    loadDataToDevice(&t_s[0].p[0].x, &t_c[0].p[0].x, t_s.size(), &polyPairs[0].is, polyPairs.size());
#endif

	//cout<<"print triangles:"<<endl;
	//printTrgl(t_s[16546]);
	//printTrgl(t_c[88008]);

#if PARALLEL_ON
	float* points;
	vtkIdType* cells;
	int nCells;
	//int nPts;
	//cout<<"**0"<<endl;
	/*
	runKernel(points, cells, nCells, nPts, _nBlock);
	writePolygonFileFastArray("data/CAM_0_small_clipped_parallel.vtk", points, cells, nCells, nPts);
	*/
	clock_t t2 = clock();
#else
	//cout<<"print triangle s from serial:"<<endl;
	//printTrgl(t_s[16546]);
	//cout<<"print triangle c from serial:"<<endl;
	//printTrgl(t_s[88008]);
    for(int i = 0; i < polyPairs.size(); i++)
    {
        if((i % 100000) == 0)
			cout<<"i = "<<i<<endl;
		//cout <<"is = "<<polyPairs[i].is<<endl;
		//cout <<"ic = "<<polyPairs[i].ic<<endl;
		vector<float2> clipped;
        clipped = clip_serial(t_s[polyPairs[i].is], t_c[polyPairs[i].ic]);
		//if(clipped.size()> 0)
		//{

		//}
		//cout<<"size = "<<clipped.size()<<endl;

		//printPolygon(clipped);
		//cout<<"is:"<<polyPairs[i].is<<endl;
		//cout<<"ic:"<<polyPairs[i].ic<<endl;
		//exit(3);

        if(clipped.size()>0)	//polygon has at least 3 edges
            clippedAll.push_back(clipped);
		//if(clippedAll.size() == 264924)
		//{
		//	cout<<"the i:"<<i << endl;
		//	break;
		//}

    }

	clock_t t2 = clock();
	/*writePolygonFileFast("data/CAM_0_small_clipped.vtk", clippedAll);*/
	vtkIdTypeArray *cellIdx = vtkIdTypeArray::New();
	
	
	for(int i = 0; i < clippedAll.size(); i++)
	{
		int np = clippedAll[i].size();
		idx_array.push_back(np);
		for(int p = 0; p < np; p++, nPts++)
		{
			idx_array.push_back(nPts);
			pts_vec.push_back(clippedAll[i][p].x);
			pts_vec.push_back(clippedAll[i][p].y);
			pts_vec.push_back(0);
		}
	}
	//points = &pts_vec[0];
	//cells = &idx_array[0];
	//nCells = clippedAll.size();
	//nPts = nPts;
#endif
//    unsigned long compute_time = (t2 - t1) * 1000 / CLOCKS_PER_SEC;
//	cout<<"Clipping time:"<< (float)compute_time * 0.001 << "sec" << endl;
  //  return clippedAll;
}

void CheckVTKFiles()
{
    string fileDir = FILE_DIR;
    string filename1 = fileDir.append("CAM_0_small_clipped.vtk");//"/home/xtong/data/VTK-LatLon2/CAM_1_vec.vtk";
    string filename2 = fileDir.append("CAM_0_small_clipped_parallel.vtk");//"data/CAM_0_small_clipped_parallel.vtk";//"/home/xtong/data/VTK-LatLon2/CAM_1_vec_warped_5times.vtk";


    vtkSmartPointer<vtkUnstructuredGridReader> reader1 =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();
    /************constraint polygon****************/
    reader1->SetFileName(filename1.c_str());
    reader1->Update(); // Needed because of GetScalarRange
    vtkUnstructuredGrid* grid1 = reader1->GetOutput();
    vtkPoints* points1 = grid1->GetPoints();
    vtkCellArray* cell1 = grid1->GetCells();
	//reader->CloseVTKFile();

	vtkSmartPointer<vtkUnstructuredGridReader> reader2 =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader2->SetFileName(filename2.c_str());
    reader2->Update();
    vtkUnstructuredGrid* grid2 = reader2->GetOutput();
    vtkPoints* points2 = grid2->GetPoints();
    vtkCellArray* cell2 = grid2->GetCells();

	cout<<cell1->GetNumberOfCells()<<","<<cell2->GetNumberOfCells()<<"]"<<endl;

		vtkNew<vtkIdList> pts1;
	vtkNew<vtkIdList> pts2;
	int size = cell1->GetData()->GetDataSize();
	vtkIdTypeArray *idArray1 = cell1->GetData();
	vtkIdTypeArray *idArray2 = cell2->GetData();
	cout<<"size of cell array:"<<size<<endl;
	for(int i = 0; i < size; i++)
	{
		//vtkSmartPointer<vtkIdList> pts1 = vtkSmartPointer<vtkIdList>::New();
		//vtkSmartPointer<vtkIdList> pts2 = vtkSmartPointer<vtkIdList>::New();

	///	cout<<cell1->GetNumberOfCells()<<endl;
		//cell1->GetCell(i, pts1.GetPointer());
		//cell2->GetCell(i, pts2.GetPointer());
		
		if(i % 10000 == 0)
			cout<<i<<endl;
			//cout<<pts1->GetId(0)<<",";
			//cout<<pts2->GetId(0)<<endl;
		if(idArray1->GetValue(i) != idArray2->GetValue(i))
		{
			cout<<"diff point:"<<i<<",";
			cout<<idArray1->GetValue(i+ 1)<<endl;
			exit(1);
		}
	}

	reader1->CloseVTKFile();
	reader2->CloseVTKFile();
}



int main( int argc, char *argv[] )
{
    //CheckVTKFiles();
    string fileDir = FILE_DIR;
    string filename_constraint = FILE_DIR;
    string filename_subject = FILE_DIR;
    filename_constraint.append("CAM_1_vec.vtk");
    filename_subject.append("CAM_1_vec_warped_5times.vtk");
//    filename_constraint.append("CAM_1_vec_resampled_warped.vtk");
//    filename_subject.append("CAM_1_vec_resampled.vtk");

    cout<<"filename_constraint:"<<filename_constraint <<endl;
    cout<<"filename_subject:"<<filename_subject <<endl;

    //0.01 is the optimal step size for original data without resampling
    float binStep = 0.01;

	if(argc > 1)
		_nBlock = strtol(argv[1], NULL, 10);
	else
		_nBlock = 512;

	if(argc > 2)
		binStep = ::atof(argv[2]);

    _t0 = clock();

	cout<<"CUDA block size: "<<_nBlock<<endl;
	cout<<"Size of Bin (radian): "<<binStep<<endl;


#if PARALLEL_ON
	float* points;
	vtkIdType* cells;
	int nCells;
	int nPts;
    initCUDA();

    PrintElapsedTime("initiate CUDA");

    runCUDA(filename_subject.c_str(), filename_constraint.c_str(), binStep, points, cells, nCells, nPts, _nBlock);
#else
	clock_t t1 = clock();
    vector<triangle> trias_c;
    vector<triangle> trias_s;
    

    /************constraint polygon****************/
	vtkSmartPointer<vtkUnstructuredGridReader> reader =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();

    reader->SetFileName(filename_constraint);
    reader->Update(); // Needed because of GetScalarRange
    vtkUnstructuredGrid* grid_c = reader->GetOutput();
    vtkPoints* points_c = grid_c->GetPoints();
    vtkCellArray* cell_c = grid_c->GetCells();
    cout<<"nc="<<cell_c->GetNumberOfCells()<<endl;
    //reader->CloseVTKFile();

    ImportTriangles(points_c, cell_c, trias_c);
    /************subject polygon****************/

	vtkSmartPointer<vtkUnstructuredGridReader> reader2 =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader2->SetFileName(filename_subject);
    reader2->Update();
    vtkUnstructuredGrid* grid_s = reader2->GetOutput();
    vtkPoints* points_s = grid_s->GetPoints();
    vtkCellArray* cell_s = grid_s->GetCells();
//	reader2->CloseVTKFile();


    ImportTriangles(points_s, cell_s, trias_s);
    vector<vector<int> > cellsInBin = Binning(trias_c);
	vector<float> pts_vec;
	vector<vtkIdType> idx_array;
	int nPts = 0;
	clipSets(trias_s, trias_c, cellsInBin, 	pts_vec, idx_array, nPts);

#endif
    PrintElapsedTime("Entire computing time");

#if PARALLEL_ON
	writePolygonFileFastArray("data/clipped_parallel.vtk", points, cells, nCells, nPts);
#else
	//cout<<"ncell, npts"<<idx_array.size()<<","<<nPts<<endl;
	//cout<<"idx_array.size():"<<idx_array.size()<<endl;
	writePolygonFileFastArray("data/clipped_serial.vtk", &pts_vec[0], &idx_array[0], idx_array.size() - nPts, nPts);
#endif

    PrintElapsedTime("write file");
 //   writePolygonFile("data/CAM_0_small_clipped.vtk", clippedPoly);
    return 1;
}
