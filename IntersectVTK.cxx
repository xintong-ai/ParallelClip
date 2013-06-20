#include "vtkActor.h"
#include "vtkCamera.h"
#include "vtkGeometryFilter.h"
#include "vtkNew.h"
#include "vtkPolyDataMapper.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkTestUtilities.h"
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkUnstructuredGridWriter.h"
#include <vtkSmartPointer.h>
#include <iostream>
#include "vtkCellArray.h"
#include "time.h"
using namespace std;

#define STEP_X 1
#define STEP_Y 1
#define X_MAX 360
#define X_MIN 0
#define Y_MAX 90
#define Y_MIN -90
#define TEST_QUAD 0

static int rangeX = X_MAX - X_MIN;
static int rangeY = Y_MAX - Y_MIN;
static int nbinX = rangeX / STEP_X;
static int nbinY = rangeY / STEP_Y;
static int nbin = nbinX * nbinY;
static int cnt = 0;
//static int threshX = rangeX / nbinX;

struct point
{
    float x;
    float y;
    point(float _x, float _y)
    {
        x = _x;
        y = _y;
    }
    point(){};
};

#if TEST_QUAD
struct quad
{
    point p[4];
    quad(point p0, point p1, point p2, point p3)
    {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
        p[3] = p3;
    }
    quad(point _p[4])
    {
        p[0] = _p[0];
        p[1] = _p[1];
        p[2] = _p[2];
        p[3] = _p[3];
    }
};
#else
struct triangle
{
    point p[3];
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
#endif

typedef struct _node
{
  float x, y;
  struct _node *next;
  struct _node *prev;
  struct _node *nextPoly;   /* pointer to the next polygon */
  struct _node *neighbor;   /* the coresponding intersection point */
  int intersect;            /* 1 if an intersection point, 0 otherwise */
  int entry;                /* 1 if an entry point, 0 otherwise */
  int visited;              /* 1 if the node has been visited, 0 otherwise */
  float alpha;              /* intersection point placemet */
} node;


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

inline int GetBin(double x, double y)
{
    return GetYBin(y) * nbinX + GetXBin(x);
}

inline int GetBin(double x, double y, int &ix, int &iy)
{
    ix = GetXBin(x);
    iy = GetYBin(y);
}

#if TEST_QUAD
inline int min(int x1, int x2, int x3, int x4)
{
    int xmin;
    if(x1 < x2)
        xmin = x1;
    else
        xmin = x2;

    if(x2 < xmin)
        xmin = x2;
    if(x3 < xmin)
        xmin = x3;
    if(x4 < xmin)
        xmin = x4;

    return xmin;
}

inline int max(int x1, int x2, int x3, int x4)
{
    int xmax;
    if(x1 > x2)
        xmax = x1;
    else
        xmax = x2;

    if(x2 > xmax)
        xmax = x2;
    if(x3 > xmax)
        xmax = x3;
    if(x4 > xmax)
        xmax = x4;

    return xmax;
}
#else

inline int min(int x1, int x2, int x3)
{
    int xmin;
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

inline int max(int x1, int x2, int x3)
{
    int xmax;
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

#endif

#if TEST_QUAD
vector<int> GetBinQuad(float q[4][2])
{
    vector<int> ret;
    int xmin, xmax, ymin, ymax;
    xmin = min(GetXBin(q[0][0]), GetXBin(q[1][0]),
            GetXBin(q[2][0]), GetXBin(q[3][0]));
    xmax = max(GetXBin(q[0][0]), GetXBin(q[1][0]),
            GetXBin(q[2][0]), GetXBin(q[3][0]));
    ymin = min(GetYBin(q[0][1]), GetYBin(q[1][1]),
            GetYBin(q[2][1]), GetYBin(q[3][1]));
    ymax = max(GetYBin(q[0][1]), GetYBin(q[1][1]),
            GetYBin(q[2][1]), GetYBin(q[3][1]));
    //ignore the posibility that for a quad,
    //some vertices is on one side of longitude
    //and some other vertices are on the other side
    if((xmax - xmin) > 10 || (ymax - ymin) > 10)
        return ret;
    for(int x = xmin; x < xmax; x++)
    {
        for(int y = ymin; y < ymax; y++)
        {
            ret.push_back(y * nbinX + x);
        }
    }
    return ret;
}

vector<int> GetBinQuad(quad q)
{
    vector<int> ret;
    int xmin, xmax, ymin, ymax;
    xmin = min(
            GetXBin(q.p[0].x),
            GetXBin(q.p[1].x),
            GetXBin(q.p[2].x),
            GetXBin(q.p[3].x));
    xmax = max(
            GetXBin(q.p[0].x),
            GetXBin(q.p[1].x),
            GetXBin(q.p[2].x),
            GetXBin(q.p[3].x));
    ymin = min(
            GetYBin(q.p[0].y),
            GetYBin(q.p[1].y),
            GetYBin(q.p[2].y),
            GetYBin(q.p[3].y));
    ymax = max(
            GetYBin(q.p[0].y),
            GetYBin(q.p[1].y),
            GetYBin(q.p[2].y),
            GetYBin(q.p[3].y));
    //if the two points are too far away, ignore it
    if((ymax - ymin) > (nbinX / 2) || (xmax - xmin) > (nbinY / 2))
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
#else

vector<int> GetBinTriangle(triangle q)
{
    vector<int> ret;
    int xmin, xmax, ymin, ymax;
    //value can be out of boundary, so mode the number of bins in each coordinate
    xmin = min(
            GetXBin(q.p[0].x),
            GetXBin(q.p[1].x),
            GetXBin(q.p[2].x));
    xmax = max(
            GetXBin(q.p[0].x),
            GetXBin(q.p[1].x),
            GetXBin(q.p[2].x));
    ymin = min(
            GetYBin(q.p[0].y),
            GetYBin(q.p[1].y),
            GetYBin(q.p[2].y));
    ymax = max(
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
#endif

#if TEST_QUAD
vector<vector<int> > Binning(vector<quad> q)
{
    vector<vector<int> > CellsInBin(nbin, vector<int>(0));
    for(int i = 0; i < q.size(); i++)
    {
        vector<int> bins = GetBinQuad(q[i]);
        for(int b = 0; b < bins.size(); b++)
        {
            CellsInBin[bins[b]].push_back(i);
        }
    }
    return CellsInBin;
}
#else

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
#endif

#if TEST_QUAD
//import vtk format to the local vector format
void ImportQuads(vtkPoints* vtkPts, vtkCellArray* vtkCls, vector<quad> &quads)
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
        quad q(p);
        quads.push_back(q);
    }
}
#else

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
#endif

#if TEST_QUAD
void InitNodes(quad q, node* &sc)
{

    for(int i = 0; i < 4; i++)
    {
        node *newNode;


        newNode = (node*)malloc(sizeof(node));
        newNode->x = q.p[i].x;
        newNode->y = q.p[i].y;
        newNode->prev = 0;        /* not need to initialize with 0 after malloc ... */
        newNode->nextPoly = 0;
        newNode->neighbor = 0;
        newNode->intersect = 0;
        newNode->entry = 0;
        newNode->visited = 0;
        newNode->alpha = 0.;
        newNode->next = sc;
        if (sc)
            sc->prev = newNode;
        sc = newNode;
    }
}
#else

void InitNodes(triangle q, node* &sc)
{

    for(int i = 0; i < 3; i++)
    {
        node *newNode;
        newNode = (node*)malloc(sizeof(node));
        newNode->x = q.p[i].x;
        newNode->y = q.p[i].y;
        newNode->prev = 0;        /* not need to initialize with 0 after malloc ... */
        newNode->nextPoly = 0;
        newNode->neighbor = 0;
        newNode->intersect = 0;
        newNode->entry = 0;
        newNode->visited = 0;
        newNode->alpha = 0.;
        newNode->next = sc;
        if (sc)
            sc->prev = newNode;
        sc = newNode;
    }
}
#endif

void view_node(node *p)
{
  if(p) printf("%c%c%c (%3d,%3d)  %f    c:%10p n:%10p P:%10p\n",
        p->intersect ? 'I' : ' ',
        p->entry ? 'E' : ' ',
        p->visited ? 'X' : ' ',
        p->x, p->y, p->alpha, p, p->neighbor, p->nextPoly);
  else  puts("NULL");
}

void view(node *p)
{
  node *aux=p;
  puts("");

  if(aux) do
  {
        view_node(aux);
        aux=aux->next;
  }
  while(aux && aux != p);
}


void deleteNode(node *p)
{
  node *aux, *hold;

  if(hold=p) do
  {
        aux=p;
        p=p->next;
        free(aux);
  }
  while(p && p!=hold);
}

void insert(node *ins, node *first, node *last)
{
  node *aux=first;
  //special case need to be taken care of when inserted intersection
  //point is the same as the vertex point
  //MOD-TONG-FROM
  //while(aux != last && aux->alpha < ins->alpha) aux = aux->next;
  //MOD-TONG-TO
  while(aux != last && aux->alpha <= ins->alpha) aux = aux->next;
  //MOD-TONG-END
  ins->next = aux;
  ins->prev = aux->prev;
  ins->prev->next = ins;
  ins->next->prev = ins;
}

node *create(float x, float y, node *next, node *prev, node *nextPoly,
  node *neighbor, int intersect, int entry, int visited, float alpha)
{
  node *newNode = (node*)malloc(sizeof(node));
  newNode->x = x;
  newNode->y = y;
  newNode->next = next;
  newNode->prev = prev;
  if(prev) newNode->prev->next = newNode;
  if(next) newNode->next->prev = newNode;
  newNode->nextPoly = nextPoly;
  newNode->neighbor = neighbor;
  newNode->intersect = intersect;
  newNode->entry = entry;
  newNode->visited = visited;
  newNode->alpha = alpha;
  return newNode;
}

node *next_node(node *p)
{
  node *aux=p;
  while(aux && aux->intersect) aux=aux->next;
  return aux;
}



node *last_node(node *p)
{
  node *aux=p;
  if(aux)
      while(aux->next)
          aux=aux->next;
  return aux;
}

inline void FreeList(node * head)
{
    if (!head)
        return;
    node* next = head->next;              // start at the head.
    node* temp;
    while(next && (next != head) )      // traverse entire list. &&
    {
        temp = next;          // save node pointer.
        next = next->next;     // advance to next.
        free(temp);            // free the saved one.
    }
    free(head);
}

node *first(node *p)
{
  node *aux=p;

  if (aux)
  do aux=aux->next;
  while(aux!=p && (!aux->intersect || aux->intersect && aux->visited));
  return aux;
}

void circle(node *p)
{
  node *aux = last_node(p);
  aux->prev->next = p;
  p->prev = aux->prev;
  free(aux);
}

float dist(float x1, float y1, float x2, float y2)
{
  return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

int I(node *p1, node *p2, node *q1, node *q2,
  float *alpha_p, float *alpha_q, float *xint, float *yint)
{
  float x, y, tp, tq, t, par;

  par = (float) ((p2->x - p1->x)*(q2->y - q1->y) -
                 (p2->y - p1->y)*(q2->x - q1->x));

  if (!par) return 0;                               /* parallel lines */

  tp = ((q1->x - p1->x)*(q2->y - q1->y) - (q1->y - p1->y)*(q2->x - q1->x))/par;
  tq = ((p2->y - p1->y)*(q1->x - p1->x) - (p2->x - p1->x)*(q1->y - p1->y))/par;

  if(tp<0 || tp>1 || tq<0 || tq>1) return 0;

  x = p1->x + tp*(p2->x - p1->x);
  y = p1->y + tp*(p2->y - p1->y);

  *alpha_p = dist(p1->x, p1->y, x, y) / dist(p1->x, p1->y, p2->x, p2->y);
  *alpha_q = dist(q1->x, q1->y, x, y) / dist(q1->x, q1->y, q2->x, q2->y);
  *xint = x;
  *yint = y;

  return 1;
}

int test(node *point, node *p)
{
  node *aux, *left, i;
  int type=0;

  left = create(0, point->y, 0, 0, 0, 0, 0, 0, 0, 0.);
  for(aux=p; aux->next; aux=aux->next)
  if(I(left, point, aux, aux->next, &i.alpha, &i.alpha, &i.x, &i.y)) type++;
  free(left);
  return type%2;
}

#if TEST_QUAD
vector<vector<point> > clip(quad q_s, quad q_c)
{
    vector<vector<point> > clipped;
    /*
    vector<point> onePoly;
    for(int i = 0; i < 4; i++)
        onePoly.push_back(s.p[i]);
    clipped.push_back(onePoly);
    */

    node *s=0, *c=0, *root=0;
    int pS=1, pC=1;


    InitNodes(q_s, s);
    InitNodes(q_c, c);


    node *auxs, *auxc, *is, *ic;
    float xi, yi;
    int e;
    float alpha_s, alpha_c;

    node *crt, *newNode, *old;
    int forward;

    auxs = last_node(s);
    create(s->x, s->y, 0, auxs, 0, 0, 0, 0, 0, 0.);
    auxc = last_node(c);
    create(c->x, c->y, 0, auxc, 0, 0, 0, 0, 0, 0.);

    for(auxs = s; auxs->next; auxs = auxs->next)
    if(!auxs->intersect)
    for(auxc = c; auxc->next; auxc = auxc->next)
    if(!auxc->intersect)
    if(I(auxs, next_node(auxs->next), auxc, next_node(auxc->next),
          &alpha_s, &alpha_c, &xi, &yi))
    {
          is = create(xi, yi, 0, 0, 0, 0, 1, 0, 0, alpha_s);
          ic = create(xi, yi, 0, 0, 0, 0, 1, 0, 0, alpha_c);
          is->neighbor = ic;
          ic->neighbor = is;
          insert(is, auxs, next_node(auxs->next));
          insert(ic, auxc, next_node(auxc->next));
    }

    e = test(s, c);
    if(pS) e = 1-e;
    for(auxs = s; auxs->next; auxs = auxs->next)
    if(auxs->intersect)
    {
          auxs->entry = e;
          e = 1-e;
    }

    e=test(c, s);
    if(pC) e = 1-e;
    for(auxc = c; auxc->next; auxc = auxc->next)
    if(auxc->intersect)
    {
          auxc->entry = e;
          e = 1-e;
    }

    circle(s);
    circle(c);
    while ((crt = first(s)) != s)
    {
          old = 0;
          for(; !crt->visited; crt = crt->neighbor)
          for(forward = crt->entry ;; )
          {
                  newNode = create(crt->x, crt->y, old, 0, 0, 0, 0, 0, 0, 0.);
                  old = newNode;
                  crt->visited = 1;
                  crt = forward ? crt->next : crt->prev;
                  if(crt->intersect)
                  {
                          crt->visited = 1;
                          break;
                  }
          }

          old->nextPoly = root;
          root = old;
    }

    view(s);
    view(c);


    return clipped;



}
#else

vector<point> clip(triangle t_s, triangle t_c)
{
 //   cout<<"cnt:"<<cnt++<<endl;
    vector<point> clipped;

    /*
    vector<point> onePoly;
    for(int i = 0; i < 4; i++)
        onePoly.push_back(s.p[i]);
    clipped.push_back(onePoly);
    */

    node *s=0, *c=0, *root=0;
    int pS=1, pC=1;


    InitNodes(t_s, s);
    InitNodes(t_c, c);

    node *auxs, *auxc, *is, *ic;
    float xi, yi;
    int e;
    float alpha_s, alpha_c;

    node *crt, *newNode, *old;
    int forward;

    auxs = last_node(s);
    create(s->x, s->y, 0, auxs, 0, 0, 0, 0, 0, 0.);
    auxc = last_node(c);
    create(c->x, c->y, 0, auxc, 0, 0, 0, 0, 0, 0.);

    for(auxs = s; auxs->next; auxs = auxs->next)
    if(!auxs->intersect)
    for(auxc = c; auxc->next; auxc = auxc->next)
    if(!auxc->intersect)
    if(I(auxs, next_node(auxs->next), auxc, next_node(auxc->next),
          &alpha_s, &alpha_c, &xi, &yi))
    {
          is = create(xi, yi, 0, 0, 0, 0, 1, 0, 0, alpha_s);
          ic = create(xi, yi, 0, 0, 0, 0, 1, 0, 0, alpha_c);
          is->neighbor = ic;
          ic->neighbor = is;
          insert(is, auxs, next_node(auxs->next));
          insert(ic, auxc, next_node(auxc->next));
    }

    e = test(s, c);
    if(pS) e = 1-e;
    for(auxs = s; auxs->next; auxs = auxs->next)
    if(auxs->intersect)
    {
          auxs->entry = e;
          e = 1-e;
    }

    e=test(c, s);
    if(pC) e = 1-e;
    for(auxc = c; auxc->next; auxc = auxc->next)
    if(auxc->intersect)
    {
          auxc->entry = e;
          e = 1-e;
    }

    circle(s);
    circle(c);
    while ((crt = first(s)) != s)
    {
          old = 0;
          for(; !crt->visited; crt = crt->neighbor)
          for(forward = crt->entry ;; )
          {
                  newNode = create(crt->x, crt->y, old, 0, 0, 0, 0, 0, 0, 0.);
                  old = newNode;
                  crt->visited = 1;
                  crt = forward ? crt->next : crt->prev;
                  if(crt->intersect)
                  {
                          crt->visited = 1;
                          break;
                  }
          }

          old->nextPoly = root;
          root = old;
    }

//    view(s);
 //   view(c);


    if(root)
    {
        for(node *aux = root; aux; aux = aux->next)
        {
            point p(aux->x, aux->y);
            clipped.push_back(p);
        }
    }
    FreeList(s);
    FreeList(c);
    FreeList(root);
    return clipped;
}
#endif

#if TEST_QUAD
//clip two set of cells
vector<vector<point> > clipSets(vector<quad> q_s, vector<quad> q_c, vector<vector<int> > cellInBin_c)
{
    vector<vector<point> > clippedAll;
    //for each quad in subject set
    for(int is = 0; is < q_s.size(); is++)
    {
        quad s = q_s[is];// one quad in subject set
        vector<int> bin_s =  GetBinQuad(s); //the bins the subject quad belong to
        //for each intersected bin
        for(int ibs = 0; ibs < bin_s.size(); ibs++)
        {
            int b_s = bin_s[ibs];//bin number
            vector<int> cellIdx_c = cellInBin_c[b_s];//indices of all the constaint quads in this bin
            for(int ic = 0; ic < cellIdx_c.size(); ic++)
            {
                vector<vector<point> > clipped = clip(q_c[ic], s);
                for(int i = 0; i < clipped.size(); i++)
                    clippedAll.push_back(clipped[i]);
            }
        }
#if 0
        vector<point> onePoly;
        for(int i = 0; i < 4; i++)
            onePoly.push_back(s.p[i]);
        clippedAll.push_back(onePoly);
#endif
        if((is % 100) == 0)
            cout<<"is = "<<is<<endl;
    }
    return clippedAll;
}
#else

//clip two set of cells
vector<vector<point> > clipSets(vector<triangle> t_s, vector<triangle> t_c, vector<vector<int> > cellInBin_c)
{
    vector<vector<point> > clippedAll;
    int ic;
    //for each quad in subject set
    for(int is = 0; is < t_s.size(); is++)
    {
     //   is = 7162;
        triangle s = t_s[is];// one quad in subject set
        vector<int> bin_s =  GetBinTriangle(s); //the bins the subject quad belong to

        //for each intersected bin
        for(int ibs = 0; ibs < bin_s.size(); ibs++)
        {
            int b_s = bin_s[ibs];//bin number
            vector<int> cellIdx_c = cellInBin_c[b_s];//indices of all the constaint quads in this bin
        //    cout<<"number of cellIdx_c:"<<cellIdx_c.size()<<endl;
            for(int i = 0; i < cellIdx_c.size(); i++)
            {
                ic = cellIdx_c[i];

       //         ic = 6596;
                vector<point> clipped = clip(t_c[ic], s);
                if(clipped.size()>0)
                    clippedAll.push_back(clipped);
            }
        }
#if 0
        vector<point> onePoly;
        for(int i = 0; i < 3; i++)
            onePoly.push_back(s.p[i]);
        clippedAll.push_back(onePoly);
#endif
        if((is % 1000) == 0)
            cout<<"is = "<<is<<endl;
    }
    return clippedAll;
}
#endif

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
    //char filename_constraint[100] = "/home/xtong/data/VTK-LatLon2/CAM_0_small.vtk";
    //char filename_subject[100] = "/home/xtong/data/VTK-LatLon2/CAM_0_small_vec_warped_5times.vtk";
    char filename_constraint[100] = "/home/xtong/data/VTK-LatLon2/CAM_1_vec.vtk";
    char filename_subject[100] = "/home/xtong/data/VTK-LatLon2/CAM_1_vec_warped_5times.vtk";
#if TEST_QUAD
    vector<quad> quads_c;
    vector<quad> quads_s;
#else
    vector<triangle> trias_c;
    vector<triangle> trias_s;
#endif
    vtkSmartPointer<vtkUnstructuredGridReader> reader =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();

    /************constraint polygon****************/
    reader->SetFileName(filename_constraint);
    reader->Update(); // Needed because of GetScalarRange
    vtkUnstructuredGrid* grid_c = reader->GetOutput();
    vtkPoints* points_c = grid_c->GetPoints();
    vtkCellArray* cell_c = grid_c->GetCells();
//    cout<<"nc="<<cell_c->GetNumberOfCells()<<endl;
    reader->CloseVTKFile();
#if TEST_QUAD
    ImportQuads(points_c, cell_c, quads_c);
#else
    ImportTriangles(points_c, cell_c, trias_c);
#endif
    /************subject polygon****************/

    reader->SetFileName(filename_subject);
    reader->Update();
    vtkUnstructuredGrid* grid_s = reader->GetOutput();
    vtkPoints* points_s = grid_s->GetPoints();
    vtkCellArray* cell_s = grid_s->GetCells();
    clock_t t1 = clock();

#if TEST_QUAD
    ImportQuads(points_s, cell_s, quads_s);
#else
    ImportTriangles(points_s, cell_s, trias_s);
#endif

    clock_t t2 = clock();

#if TEST_QUAD
    vector<vector<int> > cellsInBin = Binning(quads_c);
    vector<vector<point> > clippedPoly = clipSets(quads_s, quads_c, cellsInBin);
#else
    vector<vector<int> > cellsInBin = Binning(trias_c);
    vector<vector<point> > clippedPoly = clipSets(trias_s, trias_c, cellsInBin);
#endif
    clock_t t3 = clock();
    unsigned long compute_time = (t3 - t2) * 1000 / CLOCKS_PER_SEC;
    cout<<"computing time:"<< (float)compute_time * 0.001 << "sec" << endl;

    writePolygonFile("CAM_0_small_clipped.vtk", clippedPoly);
    return 1;
}
