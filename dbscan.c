#include <stdio.h>
#include <stdlib.h>

#include "dbscan.h"

// Don't use pow for performance optimization
// Get the squared distance of two points to then be compared with epsilon squared
static double get_squared_distance(double x1, double y1, double x2, double y2) {
  double dx = x1 - x2;
  double dy = y1 - y2;
  
  return (dx * dx) + (dy * dy);
}

static void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Simple custom quick sort implementation to sort the array based on the single coordinates
static void quick_sort(int *array, int start, int end, double *coords_array) {
  int pivot = end;

  if (start >= end) {
    return;
  }

  int i, j;
  for (i = start - 1, j = start; j <= pivot; j++) {
    if (coords_array[array[j]] > coords_array[array[pivot]]) {
      continue;
    }

    i++;

    swap(&array[i], &array[j]);
  }

  quick_sort(array, start, i - 1, coords_array);
  quick_sort(array, i + 1, end, coords_array);

}

static void search_kdtree(Node *node, int target_id, double *x_coords, double *y_coords, double epsilon, int depth, int *neighbors, int *num_neighbors) {
  // Base case
  if (node == NULL) {
    return;
  }

  // get the squared distance from the point we want to find the neighbors and the current point we are checking
  double squared_distance = get_squared_distance(x_coords[target_id], y_coords[target_id], x_coords[node->index], y_coords[node->index]);

  double epsilon_squared = epsilon * epsilon;

  // Compare the distance and in case add to the neighbor array and increase the count
  if (squared_distance <= epsilon_squared) {
    neighbors[*num_neighbors] = node->index;
    (*num_neighbors)++;
  }

  // 0 means X-axis and 1 means Y-axis
  int axis = depth % 2;

  // 0 for left and 1 for right
  int direction;
  
  // The 1D distance between the target point and the node, if it is less than epsilon we can not search the other branch
  double target_coord = (axis == 0) ? x_coords[target_id] : y_coords[target_id];
  double node_coord = (axis == 0) ? x_coords[node->index] : y_coords[node->index];
  double one_dimensional_distance = target_coord - node_coord;

  if (axis == 0) {
    if (x_coords[target_id] < x_coords[node->index]) {
      direction = 0;
    } else {
      direction = 1;
    }
  } else {
    if (y_coords[target_id] < y_coords[node->index]) {
      direction = 0;
    } else {
      direction = 1;
    }
  }

  // We check either the left or right direction (the main one), and in case also the opposite one
  if (direction == 0) {
    search_kdtree(node->left, target_id, x_coords, y_coords, epsilon, depth + 1, neighbors, num_neighbors);

    if ((one_dimensional_distance * one_dimensional_distance) <= epsilon_squared) {
      search_kdtree(node->right, target_id, x_coords, y_coords, epsilon, depth + 1, neighbors, num_neighbors);
    }
  } else {
    search_kdtree(node->right, target_id, x_coords, y_coords, epsilon, depth + 1, neighbors, num_neighbors);

    if ((one_dimensional_distance * one_dimensional_distance) <= epsilon_squared) {
      search_kdtree(node->left, target_id, x_coords, y_coords, epsilon, depth + 1, neighbors, num_neighbors);
    }
  }
}

static Node* build_kdtree(int *id_array, int start, int end, int depth, double *x_coords, double *y_coords) {
  if (start > end) {
    return NULL;
  }

  // 0 means X-axis, 1 means Y-axis
  int axis = depth % 2;

  // Sort the array based on the axis
  if (axis == 0) {
    quick_sort(id_array, start, end, x_coords);
  } else {
    quick_sort(id_array, start, end, y_coords);
  }

  // Median index of the current slice
  int median_idx = start + (end - start) / 2;

  // Node creation
  Node *node = (Node*)malloc(sizeof(Node));
  node->index = id_array[median_idx];

  // Recursion, the left child gets everything before the median while the right child everything after
  node->left = build_kdtree(id_array, start, median_idx - 1, depth + 1, x_coords, y_coords);
  node->right = build_kdtree(id_array, median_idx + 1, end, depth + 1, x_coords, y_coords);

  return node;
}

static void free_kdtree(Node *node) {
    if (node == NULL) {
        return;
    }
    
    // 1. Destroy the left branch first
    free_kdtree(node->left);
    
    // 2. Destroy the right branch second
    free_kdtree(node->right);
    
    // 3. Now that the children are gone, destroy the parent
    free(node);
}

void run_dbscan(int n, double *x_coords, double *y_coords, double epsilon, int min_pts, int *labels) {
  // Create an array of IDs from 0 to n-1
  int* id_array = (int*)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    id_array[i] = i;
  }

  printf("Starting to build tree...\n");
  Node *root = build_kdtree(id_array, 0, n - 1, 0, x_coords, y_coords);
  printf("Tree built successfully! Starting DBSCAN...\n");



  int *visited = (int*)calloc(n, sizeof(int));
  int *inqueue = (int*)calloc(n, sizeof(int));

  int *queue = (int*)malloc(n * sizeof(int));
  int *neighbors = (int*)malloc(n * sizeof(int));

  int depth = 0;
    
  int current_cluster_id = 0;

  // Loop through all the points
  for (int i = 0; i < n; i++) {
    // Check if a point has been visited
    if (visited[i]) {
      continue;
    }

    visited[i] = 1;

    int num_of_neighbors = 0;
  
    search_kdtree(root, i, x_coords, y_coords, epsilon, depth, neighbors, &num_of_neighbors);

    // If it has not been visited we check if it is a core point
    if (num_of_neighbors < min_pts) {
      labels[i] = -1;
    } else {
      current_cluster_id++;
      labels[i] = current_cluster_id;

      int read_index = 0;
      int write_index = 0;

      // We add all the neighbor points to the queue
      for (int j = 0; j < num_of_neighbors; j++) {
        if (inqueue[neighbors[j]]) {
          continue;
        }
        queue[j] = neighbors[j];
        write_index++;
        inqueue[neighbors[j]] = 1;
      }

      // We loop through the queue until its empty and perform the same check for each point in it
      while (read_index < write_index) {
        int current_point = queue[read_index];
        read_index++;

        // A point that is in the queue is part of its cluster if it is not already been claimed from another one
        if (labels[current_point] == -1 || labels[current_point] == 0) {
          labels[current_point] = current_cluster_id;
        }
      
        // If the point hasn't been visited we check if it is a core point
        if (visited[current_point] == 0) {
          visited[current_point] = 1;
          num_of_neighbors = 0;
          search_kdtree(root, current_point, x_coords, y_coords, epsilon, depth, neighbors, &num_of_neighbors);
          if (num_of_neighbors >= min_pts) {
            // If it is a core point we push its neighbour points at the end of the queue
            for (int k = 0; k < num_of_neighbors; k++) {
              if (inqueue[neighbors[k]]) {
                continue;
              }
              queue[write_index] = neighbors[k];
              write_index++;
              inqueue[neighbors[k]] = 1;
            }
          }
        }
      }
    }
  }
  free_kdtree(root);
  free(id_array);
  free(visited);
  free(queue);
  free(inqueue);
  free(neighbors);
}

