#include <stdlib.h>

#define M 512*512
#define N 512*512

void twodfivept(double s, int n, double a[M][n], double b[M][n]) {
    for (int j = 1; j < (M - 1); ++j)
        for (int i = 1; i < (n - 1); ++i)
            b[j][i] = (a[j][i-1] + a[j][i+1] + a[j-1][i] + a[j+1][i]) * (s ? a[j][i-1] > 0 : 1.0);
}

int main(int argc, char **argv)
{
    double a[M][N];
    double b[M][N];
    
    for(int i=0; i<M; ++i)
        for(int j=0; j<N; ++j)
            a[i][j] = b[i][j] = 42.0;
  
    twodfivept(0.23, N, a, b);
    
    return 0;
}
