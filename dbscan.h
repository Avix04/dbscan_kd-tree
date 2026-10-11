#ifndef DBSCAN_H
#define DBSCAN_H

typedef struct Node{
  int index;
  struct Node *left;
  struct Node *right;
} Node;

void run_dbscan(int n, double *x_coords, double *y_coords, double epsilon, int min_pts, int *labels);

#endif