# Fast2D-DBSCAN

A **high-performance**, **2D-optimized** implementation of the DBSCAN clustering algorithm. By pairing a custom C-based KD-Tree engine with a Python frontend via `ctypes`, this project achieves $O(n \log n)$ complexity and is on par with or outpaces generalized machine learning libraries (like `scikit-learn`) for large-scale, two-dimensional spatial data.

## Key Features

- **Custom KD-Tree Search Engine**: Replaces the standard $O(n^2)$ brute-force distance calculations with recursive geometric pruning, bringing spatial queries down to $O(\log n)$.
- **Structure of Arrays (SoA) Optimization**: Accepts coordinates as separate contiguous arrays (`X` and `Y`) rather than coupled structures. This maximizes CPU L1 cache hits during tree traversal and 1D boundary checks.
- **Zero-Branching Distance Metric**: Hardcoded specifically for 2D Euclidean squared distances, eliminating the heavy dimension-iteration loops and branching found in generalized N-dimensional clustering libraries.
- **Memory-Safe**: Includes full recursive garbage collection for the KD-Tree nodes and flat arrays, ensuring zero memory leaks during execution.

# Installation & Compilation

Since the core engine is written in C, you must compile it into a shared library before running the Python script.

For Linux / MacOS:

```Bash
gcc -shared -o dbscan.so -fPIC dbscan.c -O3
```

For Windows:

```Bash
gcc -shared -o dbscan.dll dbscan.c -O3
```

(Note: The `-O3` flag enables maximum compiler optimizations for even faster execution).

To run the program, simply execute the Python script:

```Bash
python main.py
```

# Usage

The `main.py` script demonstrates the usage of the `FastDBSCAN` class, which is a Python wrapper around the C library.
