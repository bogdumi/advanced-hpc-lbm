/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <malloc.h>
#include <string.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  float* speeds0;
  float* speeds1;
  float* speeds2;
  float* speeds3;
  float* speeds4;
  float* speeds5;
  float* speeds6;
  float* speeds7;
  float* speeds8;
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, float** av_vels_total_ptr, 
               int* nprocs, int* rank, int* slicesPerRank, int* start, int* end);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, int* __restrict__ obstacles, 
            int nprocs, int rank, int slicesPerRank, int start, int end, float *sendBuf1, float *recvBuf1, float *sendBuf2, float *recvBuf2, int *t_cells);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

void swap_cells(t_speed** __restrict__ cells, t_speed** __restrict__ tmp_cells);
void gather(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, float* av_vels, float* av_vels_total, int tot_cells, int nprocs, int rank, int slicesPerRank, int start, int end);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, float** av_vels_total_ptr, int nprocs, int rank);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* __restrict__ cells, int* __restrict__ obstacles, int start, int end, int nprocs, int rank, int *tot_cells);
float av_velocity_reynolds(const t_param params, t_speed* __restrict__ cells, int* __restrict__ obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float *av_vels   = NULL, *av_vels_total = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  float *sendBuf1 = NULL, *recvBuf1 = NULL, *sendBuf2 = NULL, *recvBuf2 = NULL;
  int slicesPerRank = 0, start = 0, end = 0;
  int tot_cells = 0;

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &av_vels_total, &nprocs, &rank, &slicesPerRank, &start, &end);
  sendBuf1 = (float*)_mm_malloc(sizeof(float) * (params.nx * 9), 64);
  recvBuf1 = (float*)_mm_malloc(sizeof(float) * (params.nx * 9), 64);
  sendBuf2 = (float*)_mm_malloc(sizeof(float) * (params.nx * 9), 64);
  recvBuf2 = (float*)_mm_malloc(sizeof(float) * (params.nx * 9), 64);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++) {
    __assume_aligned(av_vels, 64);
    av_vels[tt] = timestep(params, cells, tmp_cells, obstacles, nprocs, rank, slicesPerRank, start, end, sendBuf1, recvBuf1, sendBuf2, recvBuf2, &tot_cells);
    swap_cells(&cells, &tmp_cells);
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  gather(params, cells, tmp_cells, av_vels, av_vels_total, tot_cells, nprocs, rank, slicesPerRank, start, end);
  swap_cells(&cells, &tmp_cells);

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, cells, obstacles, av_vels_total);
  }
  _mm_free(sendBuf1);
  _mm_free(sendBuf2);
  _mm_free(recvBuf1);
  _mm_free(recvBuf2);
  sendBuf1 = NULL;
  sendBuf2 = NULL;
  recvBuf1 = NULL;
  recvBuf2 = NULL;
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &av_vels_total, nprocs, rank);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

void swap_cells(t_speed** __restrict__ cells, t_speed** __restrict__ tmp_cells) {
  t_speed* aux = *cells;
  *cells = *tmp_cells;
  *tmp_cells = aux;
}

float timestep(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, int* __restrict__ obstacles, int nprocs, int rank, int slicesPerRank, int start, int end, float *sendBuf1, float *recvBuf1, float *sendBuf2, float *recvBuf2, int *t_cells)
{
  float u_sqrt = 0.f;
  int tot_cells = 0;
  if (start <= params.ny - 2 && params.ny - 2 < end) {
    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
    int jj = params.ny - 2;

    __assume_aligned(cells->speeds0, 64);
    __assume_aligned(cells->speeds1, 64);
    __assume_aligned(cells->speeds2, 64);
    __assume_aligned(cells->speeds3, 64);
    __assume_aligned(cells->speeds4, 64);
    __assume_aligned(cells->speeds5, 64);
    __assume_aligned(cells->speeds6, 64);
    __assume_aligned(cells->speeds7, 64);
    __assume_aligned(cells->speeds8, 64);
    __assume((params.nx) % 128 == 0);

    #pragma vector aligned
    #pragma ivdep
    // #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      int c = ii + jj*params.nx;
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[c]
          && (cells -> speeds3[c] - w1) > 0.f
          && (cells -> speeds6[c] - w2) > 0.f
          && (cells -> speeds7[c] - w2) > 0.f)
      {
        /* increase 'east-side' densities */
        cells -> speeds1[c] += w1;
        cells -> speeds5[c] += w2;
        cells -> speeds8[c] += w2;
        /* decrease 'west-side' densities */
        cells -> speeds3[c] -= w1;
        cells -> speeds6[c] -= w2;
        cells -> speeds7[c] -= w2;
      }
    }

  }

  int toRight = 0, fromLeft = 0, toLeft = 0, fromRight = 0, leftSlice = 0, rightSlice = 0;
  MPI_Request request1send, request2send;
  MPI_Request requestrecv[2];

  // Send right, recieve left
  // Send left, recieve right
  if (rank != 0) {
    fromLeft = rank - 1;
    toLeft = rank - 1;
    leftSlice = start - 1;
  } else {
    fromLeft = nprocs - 1;
    toLeft = nprocs - 1;
    leftSlice = params.ny - 1;
  }
  if (rank != nprocs - 1) {
    toRight = rank + 1;
    fromRight = rank + 1;
    rightSlice = end;
  } else {
    toRight = 0;
    fromRight = 0;
    rightSlice = 0;
  }

  memcpy(sendBuf1 + params.nx * 0, cells -> speeds0 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 1, cells -> speeds1 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 2, cells -> speeds2 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 3, cells -> speeds3 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 4, cells -> speeds4 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 5, cells -> speeds5 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 6, cells -> speeds6 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 7, cells -> speeds7 + (end - 1) * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf1 + params.nx * 8, cells -> speeds8 + (end - 1) * params.nx, params.nx * sizeof(float));

  memcpy(sendBuf2 + params.nx * 0, cells -> speeds0 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 1, cells -> speeds1 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 2, cells -> speeds2 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 3, cells -> speeds3 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 4, cells -> speeds4 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 5, cells -> speeds5 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 6, cells -> speeds6 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 7, cells -> speeds7 + start * params.nx, params.nx * sizeof(float));
  memcpy(sendBuf2 + params.nx * 8, cells -> speeds8 + start * params.nx, params.nx * sizeof(float));

  MPI_Isend(sendBuf1, 9 * params.nx, MPI_FLOAT, toRight, 0, MPI_COMM_WORLD, &request1send);
  MPI_Isend(sendBuf2, 9 * params.nx, MPI_FLOAT, toLeft, 0, MPI_COMM_WORLD, &request2send);

  MPI_Irecv(recvBuf1, 9 * params.nx, MPI_FLOAT, fromLeft, 0, MPI_COMM_WORLD, requestrecv);
  MPI_Irecv(recvBuf2, 9 * params.nx, MPI_FLOAT, fromRight, 0, MPI_COMM_WORLD, requestrecv + 1);

  MPI_Waitall(2, requestrecv, MPI_STATUSES_IGNORE);

  memcpy(cells -> speeds0 + leftSlice * params.nx, recvBuf1 + params.nx * 0, params.nx * sizeof(float));
  memcpy(cells -> speeds1 + leftSlice * params.nx, recvBuf1 + params.nx * 1, params.nx * sizeof(float));
  memcpy(cells -> speeds2 + leftSlice * params.nx, recvBuf1 + params.nx * 2, params.nx * sizeof(float));
  memcpy(cells -> speeds3 + leftSlice * params.nx, recvBuf1 + params.nx * 3, params.nx * sizeof(float));
  memcpy(cells -> speeds4 + leftSlice * params.nx, recvBuf1 + params.nx * 4, params.nx * sizeof(float));
  memcpy(cells -> speeds5 + leftSlice * params.nx, recvBuf1 + params.nx * 5, params.nx * sizeof(float));
  memcpy(cells -> speeds6 + leftSlice * params.nx, recvBuf1 + params.nx * 6, params.nx * sizeof(float));
  memcpy(cells -> speeds7 + leftSlice * params.nx, recvBuf1 + params.nx * 7, params.nx * sizeof(float));
  memcpy(cells -> speeds8 + leftSlice * params.nx, recvBuf1 + params.nx * 8, params.nx * sizeof(float));

  memcpy(cells -> speeds0 + rightSlice * params.nx, recvBuf2 + params.nx * 0, params.nx * sizeof(float));
  memcpy(cells -> speeds1 + rightSlice * params.nx, recvBuf2 + params.nx * 1, params.nx * sizeof(float));
  memcpy(cells -> speeds2 + rightSlice * params.nx, recvBuf2 + params.nx * 2, params.nx * sizeof(float));
  memcpy(cells -> speeds3 + rightSlice * params.nx, recvBuf2 + params.nx * 3, params.nx * sizeof(float));
  memcpy(cells -> speeds4 + rightSlice * params.nx, recvBuf2 + params.nx * 4, params.nx * sizeof(float));
  memcpy(cells -> speeds5 + rightSlice * params.nx, recvBuf2 + params.nx * 5, params.nx * sizeof(float));
  memcpy(cells -> speeds6 + rightSlice * params.nx, recvBuf2 + params.nx * 6, params.nx * sizeof(float));
  memcpy(cells -> speeds7 + rightSlice * params.nx, recvBuf2 + params.nx * 7, params.nx * sizeof(float));
  memcpy(cells -> speeds8 + rightSlice * params.nx, recvBuf2 + params.nx * 8, params.nx * sizeof(float));

  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);
  __assume_aligned(tmp_cells->speeds0, 64);
  __assume_aligned(tmp_cells->speeds1, 64);
  __assume_aligned(tmp_cells->speeds2, 64);
  __assume_aligned(tmp_cells->speeds3, 64);
  __assume_aligned(tmp_cells->speeds4, 64);
  __assume_aligned(tmp_cells->speeds5, 64);
  __assume_aligned(tmp_cells->speeds6, 64);
  __assume_aligned(tmp_cells->speeds7, 64);
  __assume_aligned(tmp_cells->speeds8, 64);
  __assume((params.nx) % 128 == 0);
  __assume((params.ny) % 128 == 0);
  
  // #pragma omp simd reduction (+:u_sqrt) reduction (+:tot_cells)
  for (int jj = start; jj < end; jj++){
    #pragma vector aligned
    #pragma ivdep
    #pragma omp simd reduction (+:u_sqrt) reduction (+:tot_cells)
    for (int ii = 0; ii < params.nx; ii++){
      int c = jj*params.nx + ii;
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells -> speeds0[c] = cells -> speeds0[ii + jj*params.nx];   /* central cell, no movement */
      tmp_cells -> speeds1[c] = cells -> speeds1[x_w + jj*params.nx];  /* east */
      tmp_cells -> speeds2[c] = cells -> speeds2[ii + y_s*params.nx];  /* north */
      tmp_cells -> speeds3[c] = cells -> speeds3[x_e + jj*params.nx];  /* west */
      tmp_cells -> speeds4[c] = cells -> speeds4[ii + y_n*params.nx];  /* south */
      tmp_cells -> speeds5[c] = cells -> speeds5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells -> speeds6[c] = cells -> speeds6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells -> speeds7[c] = cells -> speeds7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells -> speeds8[c] = cells -> speeds8[x_w + y_n*params.nx]; /* south-east */
      if (obstacles[c])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        float aux[9];

        aux[1] = tmp_cells -> speeds3[c];
        aux[2] = tmp_cells -> speeds4[c];
        aux[3] = tmp_cells -> speeds1[c];
        aux[4] = tmp_cells -> speeds2[c];
        aux[5] = tmp_cells -> speeds7[c];
        aux[6] = tmp_cells -> speeds8[c];
        aux[7] = tmp_cells -> speeds5[c];
        aux[8] = tmp_cells -> speeds6[c];

        tmp_cells -> speeds1[c] = aux[1];
        tmp_cells -> speeds2[c] = aux[2];
        tmp_cells -> speeds3[c] = aux[3];
        tmp_cells -> speeds4[c] = aux[4];
        tmp_cells -> speeds5[c] = aux[5];
        tmp_cells -> speeds6[c] = aux[6];
        tmp_cells -> speeds7[c] = aux[7];
        tmp_cells -> speeds8[c] = aux[8];
      }
      else
      {
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;

        local_density += (tmp_cells -> speeds0[c] + tmp_cells -> speeds1[c] + tmp_cells -> speeds2[c] + tmp_cells -> speeds3[c] + tmp_cells -> speeds4[c] + tmp_cells -> speeds5[c] + tmp_cells -> speeds6[c] + tmp_cells -> speeds7[c] + tmp_cells -> speeds8[c]);

        /* compute x velocity component */
        float u_x = (tmp_cells -> speeds1[c] + tmp_cells -> speeds5[c] + tmp_cells -> speeds8[c] - (tmp_cells -> speeds3[c] + tmp_cells -> speeds6[c] + tmp_cells -> speeds7[c])) / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells -> speeds2[c] + tmp_cells -> speeds5[c] + tmp_cells -> speeds6[c] - (tmp_cells -> speeds4[c] + tmp_cells -> speeds7[c] + tmp_cells -> speeds8[c])) / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;
        u_sqrt += sqrt(u_sq);
        tot_cells++;

        /* directional velocity components */
        float u[NSPEEDS] __attribute__((aligned(64)));
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS] __attribute__((aligned(64)));
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1] * u[1]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2] * u[2]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3] * u[3]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4] * u[4]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5] * u[5]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6] * u[6]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7] * u[7]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8] * u[8]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));

        /* relaxation step */
        tmp_cells -> speeds0[c] = tmp_cells -> speeds0[c] + params.omega * (d_equ[0] - tmp_cells -> speeds0[c]);
        tmp_cells -> speeds1[c] = tmp_cells -> speeds1[c] + params.omega * (d_equ[1] - tmp_cells -> speeds1[c]);
        tmp_cells -> speeds2[c] = tmp_cells -> speeds2[c] + params.omega * (d_equ[2] - tmp_cells -> speeds2[c]);
        tmp_cells -> speeds3[c] = tmp_cells -> speeds3[c] + params.omega * (d_equ[3] - tmp_cells -> speeds3[c]);
        tmp_cells -> speeds4[c] = tmp_cells -> speeds4[c] + params.omega * (d_equ[4] - tmp_cells -> speeds4[c]);
        tmp_cells -> speeds5[c] = tmp_cells -> speeds5[c] + params.omega * (d_equ[5] - tmp_cells -> speeds5[c]);
        tmp_cells -> speeds6[c] = tmp_cells -> speeds6[c] + params.omega * (d_equ[6] - tmp_cells -> speeds6[c]);
        tmp_cells -> speeds7[c] = tmp_cells -> speeds7[c] + params.omega * (d_equ[7] - tmp_cells -> speeds7[c]);
        tmp_cells -> speeds8[c] = tmp_cells -> speeds8[c] + params.omega * (d_equ[8] - tmp_cells -> speeds8[c]);
      }
    }
  }

  *t_cells = tot_cells;
  return u_sqrt;
}

float av_velocity(const t_param params, t_speed* __restrict__ cells, int* __restrict__ obstacles, int start, int end, int nprocs, int rank, int *tot_cells)
{
  int    tot_c = 0;  /* no. of cells used in calculation */
  float  tot_u = 0.f;    /* accumulated magnitudes of velocity for each cell */

  /* loop over all non-blocked cells */
  // #pragma omp parallel for reduction(+:tot_u) reduction(+:tot_c)
  for (int jj = start; jj < end; jj++)
  {
    #pragma vector aligned
    #pragma ivdep
    #pragma omp simd reduction (+:tot_u) reduction(+:tot_c)
    for (int ii = 0; ii < params.nx; ii++)
    {
      int c = ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[c])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += (cells -> speeds0[c] + cells -> speeds1[c] + cells -> speeds2[c] + cells -> speeds3[c] + cells -> speeds4[c] + cells -> speeds5[c] + cells -> speeds6[c] + cells -> speeds7[c] + cells -> speeds8[c]);

        /* compute x velocity component */
        float u_x = (cells -> speeds1[c] + cells -> speeds5[c] + cells -> speeds8[c] - (cells -> speeds3[c] + cells -> speeds6[c] + cells -> speeds7[c])) / local_density;
        /* compute y velocity component */
        float u_y = (cells -> speeds2[c] + cells -> speeds5[c] + cells -> speeds6[c] - (cells -> speeds4[c] + cells -> speeds7[c] + cells -> speeds8[c])) / local_density;
        
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_c;
      }
    }
  }

  *tot_cells = tot_c;
  return tot_u;
}

float av_velocity_reynolds(const t_param params, t_speed* __restrict__ cells, int* __restrict__ obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  #pragma omp parallel for reduction(+:tot_u) reduction(+:tot_cells)
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma vector aligned
    #pragma ivdep
    #pragma omp simd reduction (+:tot_u) reduction(+:tot_cells)
    for (int ii = 0; ii < params.nx; ii++)
    {
      int c = ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[c])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += (cells -> speeds0[c] + cells -> speeds1[c] + cells -> speeds2[c] + cells -> speeds3[c] + cells -> speeds4[c] + cells -> speeds5[c] + cells -> speeds6[c] + cells -> speeds7[c] + cells -> speeds8[c]);

        /* compute x velocity component */
        float u_x = (cells -> speeds1[c] + cells -> speeds5[c] + cells -> speeds8[c] - (cells -> speeds3[c] + cells -> speeds6[c] + cells -> speeds7[c])) / local_density;
        /* compute y velocity component */
        float u_y = (cells -> speeds2[c] + cells -> speeds5[c] + cells -> speeds6[c] - (cells -> speeds4[c] + cells -> speeds7[c] + cells -> speeds8[c])) / local_density;
        
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

void gather(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, float* av_vels, float* av_vels_total, int tot_cells, int nprocs, int rank, int slicesPerRank, int start, int end) {
  int *recvCount, *displCount;
  recvCount = (int*)_mm_malloc(sizeof(int) * nprocs, 64);
  displCount = (int*)_mm_malloc(sizeof(int) * nprocs, 64);
  int sendCount[1];
  sendCount[0] = (end - start) * params.nx;
  MPI_Gather(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
  displCount[0] = 0;
  for (int i = 1; i < nprocs; i ++)
    displCount[i] = displCount[i - 1] + recvCount[i - 1];
  
  float* sendBuf;
  sendBuf = cells -> speeds0 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds0, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds1 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds1, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds2 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds2, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds3 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds3, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds4 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds4, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds5 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds5, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds6 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds6, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds7 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds7, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  sendBuf = cells -> speeds8 + start * params.nx;
  MPI_Gatherv(sendBuf, (end - start) * params.nx, MPI_FLOAT, tmp_cells -> speeds8, recvCount, displCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  int tot_cells_grid = 0;
  MPI_Reduce(av_vels, av_vels_total, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&tot_cells, &tot_cells_grid, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank == 0){
    for (int i = 0; i < params.maxIters; i++)
      av_vels_total[i] = av_vels_total[i] / (float) tot_cells_grid;
  }

  _mm_free(recvCount);
  recvCount = NULL;
  _mm_free(displCount);
  displCount = NULL;
  return; 
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, float** av_vels_total_ptr,
               int* nprocs, int* rank, int* slicesPerRank, int* start, int* end)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  

  *slicesPerRank = params -> ny / *nprocs;

  int rem = params -> ny % *nprocs;

  if (*rank < rem)
    *start = *rank * (*slicesPerRank + 1);
  else
    *start = rem * (*slicesPerRank + 1) + (*rank - rem) * *slicesPerRank;

  *end = *start + *slicesPerRank;
  if (*rank < rem)
    (*end)++;

  *cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed), 64);

  float *s0, *s1, *s2, *s3, *s4, *s5, *s6, *s7, *s8;

  s0 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s1 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s2 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s3 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s4 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s5 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s6 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s7 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s8 = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);

  (*cells_ptr) -> speeds0 = s0;
  (*cells_ptr) -> speeds1 = s1;
  (*cells_ptr) -> speeds2 = s2;
  (*cells_ptr) -> speeds3 = s3;
  (*cells_ptr) -> speeds4 = s4;
  (*cells_ptr) -> speeds5 = s5;
  (*cells_ptr) -> speeds6 = s6;
  (*cells_ptr) -> speeds7 = s7;
  (*cells_ptr) -> speeds8 = s8;

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * (params->ny * params->nx), 64);

  float *s0_, *s1_, *s2_, *s3_, *s4_, *s5_, *s6_, *s7_, *s8_;

  s0_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s1_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s2_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s3_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s4_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s5_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s6_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s7_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);
  s8_ = (float*)_mm_malloc(sizeof(float) * (params -> nx * params -> ny), 64);

  (*tmp_cells_ptr) -> speeds0 = s0_;
  (*tmp_cells_ptr) -> speeds1 = s1_;
  (*tmp_cells_ptr) -> speeds2 = s2_;
  (*tmp_cells_ptr) -> speeds3 = s3_;
  (*tmp_cells_ptr) -> speeds4 = s4_;
  (*tmp_cells_ptr) -> speeds5 = s5_;
  (*tmp_cells_ptr) -> speeds6 = s6_;
  (*tmp_cells_ptr) -> speeds7 = s7_;
  (*tmp_cells_ptr) -> speeds8 = s8_;

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density       / 9.f;
  float w2 = params->density       / 36.f;
  
  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  for (int jj = *start; jj < *end; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      int c = ii + jj*params->nx;
      (*cells_ptr) -> speeds0[c] = w0;
      (*cells_ptr) -> speeds1[c] = w1;
      (*cells_ptr) -> speeds2[c] = w1;
      (*cells_ptr) -> speeds3[c] = w1;
      (*cells_ptr) -> speeds4[c] = w1;
      (*cells_ptr) -> speeds5[c] = w2;
      (*cells_ptr) -> speeds6[c] = w2;
      (*cells_ptr) -> speeds7[c] = w2;
      (*cells_ptr) -> speeds8[c] = w2;
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);
  *av_vels_total_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, float** av_vels_total_ptr,
             int nprocs, int rank)
{
  /*
  ** free up allocated memory
  */

  _mm_free((*cells_ptr) -> speeds0);
  _mm_free((*cells_ptr) -> speeds1);
  _mm_free((*cells_ptr) -> speeds2);
  _mm_free((*cells_ptr) -> speeds3);
  _mm_free((*cells_ptr) -> speeds4);
  _mm_free((*cells_ptr) -> speeds5);
  _mm_free((*cells_ptr) -> speeds6);
  _mm_free((*cells_ptr) -> speeds7);
  _mm_free((*cells_ptr) -> speeds8);

  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free((*tmp_cells_ptr) -> speeds0);
  _mm_free((*tmp_cells_ptr) -> speeds1);
  _mm_free((*tmp_cells_ptr) -> speeds2);
  _mm_free((*tmp_cells_ptr) -> speeds3);
  _mm_free((*tmp_cells_ptr) -> speeds4);
  _mm_free((*tmp_cells_ptr) -> speeds5);
  _mm_free((*tmp_cells_ptr) -> speeds6);
  _mm_free((*tmp_cells_ptr) -> speeds7);
  _mm_free((*tmp_cells_ptr) -> speeds8);

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  _mm_free(*av_vels_total_ptr);
  *av_vels_total_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity_reynolds(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int c = ii + jj*params.nx;
      total += (cells -> speeds0[c] + cells -> speeds1[c] + cells -> speeds2[c] + cells -> speeds3[c] + cells -> speeds4[c] + cells -> speeds5[c] + cells -> speeds6[c] + cells -> speeds7[c] + cells -> speeds8[c]);
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        int c = ii + jj*params.nx;

        local_density = 0.f;
        local_density += (cells -> speeds0[c] + cells -> speeds1[c] + cells -> speeds2[c] + cells -> speeds3[c] + cells -> speeds4[c] + cells -> speeds5[c] + cells -> speeds6[c] + cells -> speeds7[c] + cells -> speeds8[c]);

        /* compute x velocity component */
        u_x = (cells -> speeds1[c] + cells -> speeds5[c] + cells -> speeds8[c] - (cells -> speeds3[c] + cells -> speeds6[c] + cells -> speeds7[c])) / local_density;
        /* compute y velocity component */
        u_y = (cells -> speeds2[c] + cells -> speeds5[c] + cells -> speeds6[c] - (cells -> speeds4[c] + cells -> speeds7[c] + cells -> speeds8[c])) / local_density;
        
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}