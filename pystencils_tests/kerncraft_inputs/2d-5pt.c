double a[30][50][3];
double b[30][50][3];
double s;

for(int j=1; j<30-1; ++j)
    for(int i=1; i<50-1; ++i)
        b[j][i] = ( a[j][i-1] + a[j][i+1]
                  + a[j-1][i] + a[j+1][i]) * s;
