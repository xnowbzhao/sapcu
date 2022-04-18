
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath> 
#include <queue>
#include <unordered_map>
#include <algorithm>
using namespace std;
int k = 3, idx;  

struct Vector3
{
    double xyz[3];
    Vector3(double x, double y, double z)
    {
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
    }
    Vector3()
    {
        xyz[0] = 0;
        xyz[1] = 0;
        xyz[2] = 0;
    }
    Vector3 operator+(const Vector3& others)
    {
        Vector3 s1;
        s1.xyz[0] = this->xyz[0] + others.xyz[0];
        s1.xyz[1] = this->xyz[1] + others.xyz[1];
        s1.xyz[2] = this->xyz[2] + others.xyz[2];
        return s1;
    }
    Vector3 operator-(const Vector3& others)
    {
        Vector3 s1;
        s1.xyz[0] = this->xyz[0] - others.xyz[0];
        s1.xyz[1] = this->xyz[1] - others.xyz[1];
        s1.xyz[2] = this->xyz[2] - others.xyz[2];
        return s1;
    }
    Vector3 operator/(const double& others)
    {
        Vector3 s1;
        s1.xyz[0] = this->xyz[0] / others;
        s1.xyz[1] = this->xyz[1] / others;
        s1.xyz[2] = this->xyz[2] / others;
        return s1;
    }
    Vector3 operator*(const double& others)
    {
        Vector3 s1;
        s1.xyz[0] = this->xyz[0] * others;
        s1.xyz[1] = this->xyz[1] * others;
        s1.xyz[2] = this->xyz[2] * others;
        return s1;
    }
    bool operator < (const Vector3& u) const
    {
        return xyz[idx] < u.xyz[idx];
    }
}po[5001];


priority_queue<pair<double, Vector3>>nq;

struct kdTree
{
    Vector3 pt[20004];
    int son[20004];

    void build(int l, int r, int rt = 1, int dep = 0)
    {
        if (l > r) return;
        son[rt] = r - l;
        son[rt * 2] = son[rt * 2 + 1] = -1;
        idx = dep % k;
        int mid = (l + r) / 2;
        nth_element(po + l, po + mid, po + r + 1);
        pt[rt] = po[mid];
        build(l, mid - 1, rt * 2, dep + 1);
        build(mid + 1, r, rt * 2 + 1, dep + 1);
    }
    void query(Vector3 p, int m, int rt = 1, int dep = 0)
    {
        if (son[rt] == -1) return;
        pair<double, Vector3> nd(0, pt[rt]);
        for (int i = 0; i < k; i++)
            nd.first += (nd.second.xyz[i] - p.xyz[i]) * (nd.second.xyz[i] - p.xyz[i]);
        int dim = dep % k, x = rt * 2, y = rt * 2 + 1, fg = 0;
        if (p.xyz[dim] >= pt[rt].xyz[dim])
            swap(x, y);
        if (~son[x])
            query(p, m, x, dep + 1);
        if (nq.size() < m)
            nq.push(nd), fg = 1;
        else
        {
            if (nd.first < nq.top().first)
                nq.pop(), nq.push(nd);
            if ((p.xyz[dim] - pt[rt].xyz[dim]) * (p.xyz[dim] - pt[rt].xyz[dim]) < nq.top().first)
                fg = 1;
        }
        if (~son[y] && fg)
            query(p, m, y, dep + 1);
    }
}kd;

double cell;
int boxsize;
int go[6][3] = { 1,0,0,
-1,0,0,
0,1,0,
0,-1,0,
0,0,1,
0,0,-1 };
double Dot(Vector3 a, Vector3 b)
{
    return a.xyz[0] * b.xyz[0] + a.xyz[1] * b.xyz[1] + a.xyz[2] * b.xyz[2];
}
double Dist(Vector3 a, Vector3 b)
{
    return    sqrt((a.xyz[0] - b.xyz[0]) * (a.xyz[0] - b.xyz[0]) + (a.xyz[1] - b.xyz[1]) * (a.xyz[1] - b.xyz[1]) + (a.xyz[2] - b.xyz[2]) * (a.xyz[2] - b.xyz[2]));
}
Vector3 Cross(Vector3 a, Vector3 b)
{
    Vector3 s1;
    s1.xyz[0] = a.xyz[1] * b.xyz[2] - a.xyz[2] * b.xyz[1];
    s1.xyz[1] = a.xyz[2] * b.xyz[0] - a.xyz[0] * b.xyz[2];
    s1.xyz[2] = a.xyz[0] * b.xyz[1] - a.xyz[1] * b.xyz[0];
    return s1;
}
Vector3 PointTri(Vector3 a, Vector3 b, Vector3 c, Vector3 p)
{
    Vector3 ab = b - a;
    Vector3 ac = c - a;
    Vector3 bc = c - b;

    double snom = Dot(p - a, ab);
    double sdenom = Dot(p - b, a - b);
    double tnom = Dot(p - a, ac);
    double tdenom = Dot(p - c, a - c);

    if (snom <= 0.0f && tnom <= 0.0f)
        return a;

    double unom = Dot(p - b, bc);
    double udenom = Dot(p - c, b - c);
    if (sdenom <= 0.0f && unom <= 0.0f)
        return b;
    if (tdenom <= 0.0f && udenom <= 0.0f)
        return c;

    Vector3 n = Cross(b - a, c - a);
    double vc = Dot(n, Cross(a - p, b - p));

    if (vc <= 0.0f && snom >= 0.0f && sdenom >= 0.0f)
        return a + ab * snom / (snom + sdenom);

    double va = Dot(n, Cross(b - p, c - p));
    if (va <= 0.0f && unom >= 0.0f && udenom >= 0.0f)
        return b + bc * unom / (unom + udenom);

    double vb = Dot(n, Cross(c - p, a - p));
    if (vb <= 0.0f && tnom >= 0.0f && tdenom >= 0.0f)
        return a + ac * tnom / (tnom + tdenom);

    double u = va / (va + vb + vc);
    double v = vb / (va + vb + vc);
    double w = 1.0f - u - v;
    return a * u + b * v + c * w;
}
int main(int argc, char** argv)
{
    queue<int> search;
    int pnumber;
    
    FILE* pointreader = fopen("test.xyz", "r");
    FILE* targetwriter = fopen("target.xyz", "w");
    unordered_map<int, double> dist;
    sscanf(argv[1], "%lf", &cell);
    sscanf(argv[2], "%d", &pnumber);
    
    boxsize = (round)(1 / cell);
    for (int i = 0; i < pnumber; i++)
    {
        fscanf(pointreader, "%lf %lf %lf", &po[i].xyz[0], &po[i].xyz[1], &po[i].xyz[2]);
        int ppp = floor(((po[i].xyz[0] + 0.5) / cell)) * boxsize * boxsize + floor(((po[i].xyz[1] + 0.5) / cell)) * boxsize + floor(((po[i].xyz[2] + 0.5) / cell));
        search.push(ppp);
    }
    kd.build(0, pnumber);
    int count = 0;
    while (!search.empty())
    {
        count++;
        int tpp = search.front();
        search.pop();
        if (dist.find(tpp) != dist.end())
        {
            continue;
        }
        int tppp = tpp;
        int z = tppp % boxsize;
        tppp /= boxsize;
        int y = tppp % boxsize;
        tppp /= boxsize;
        int x = tppp;
        Vector3 center;
        center.xyz[0] = x * cell + 0.5 * cell - 0.5;
        center.xyz[1] = y * cell + 0.5 * cell - 0.5;
        center.xyz[2] = z * cell + 0.5 * cell - 0.5;
        kd.query(center, 10);
        Vector3 pt[10];
        for (int j = 0; !nq.empty(); j++)
            pt[j] = nq.top().second, nq.pop();
        double  tdist = 99999999999999;
        for (int i = 0; i < 8; i++)
        {
            Vector3 tt = PointTri(pt[i], pt[8], pt[9], center);
            double tmepdist = Dist(tt, center);
            if (tmepdist < tdist)
                tdist = tmepdist;
        }
        if (dist.find(tpp) == dist.end())
        {
            dist[tpp] = tdist;
            if (tdist >= 0.0110 && tdist <= 0.0150)
            {
                fprintf(targetwriter, "%lf %lf %lf\n", center.xyz[0], center.xyz[1], center.xyz[2]);
            }
            else if (tdist > 0.0150)
                continue;
            for (int i = 0; i < 6; i++)
            {
                int tx = x + go[i][0];
                int ty = y + go[i][1];
                int tz = z + go[i][2];
                int ttp = tx * boxsize * boxsize + ty * boxsize + tz;
                if (dist.find(ttp) == dist.end())
                {
                    search.push(ttp);
                }
            }
        }
    }

    fclose(pointreader);
    fclose(targetwriter);
    return 0;
}

