#include "omp.h"
#include <iostream>
#include <time.h>
#include <immintrin.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace std;
using namespace chrono;

const int maxN = 1024;
const int rand_range = 100;
const int us_per_ms = 1e3;
int T;
int n;

float A[maxN][maxN];
float B[maxN][maxN];
float C[maxN][maxN];

#define ThreadNumber 4


void print_matrix(int n, float c[][maxN]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << c[i][j] << ' ';
		}
		cout << endl;
	}
}

void transpose(int n, float a[][maxN]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			swap(a[i][j], a[j][i]);
		}
	}
}

void serial_mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0.0;
			for (int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}
void serial_mul_openmp(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
    #pragma omp parallel for num_threads(ThreadNumber)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0.0;
			for (int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void trans_mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	transpose(n, b);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0.0;
			for (int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	transpose(n, b);
}

void trans_mul_openmp(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	transpose(n, b);
	#pragma omp parallel for num_threads(ThreadNumber)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0.0;
    		for (int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	transpose(n, b);
}


void sse_mul_openmp(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	__m128 t1, t2, sum;
	transpose(n, b);
    #pragma omp simd
    for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0.0;
			sum = _mm_setzero_ps();

			for (int k = n - 4; k >= 0; k -= 4) {
				t1 = _mm_loadu_ps(a[i] + k);
				t2 = _mm_loadu_ps(b[j] + k);
				t1 = _mm_mul_ps(t1, t2);
				sum = _mm_add_ps(sum, t1);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(c[i] + j, sum);
			for (int k = (n % 4) - 1; k >= 0; --k) {
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	transpose(n, b);
}

void sse_mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	__m128 t1, t2, sum;
	transpose(n, b);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0.0;
			sum = _mm_setzero_ps();
			for (int k = n - 4; k >= 0; k -= 4) {
				t1 = _mm_loadu_ps(a[i] + k);
				t2 = _mm_loadu_ps(b[j] + k);
				t1 = _mm_mul_ps(t1, t2);
				sum = _mm_add_ps(sum, t1);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(c[i] + j, sum);
			for (int k = (n % 4) - 1; k >= 0; --k) {
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	transpose(n, b);
}

float **matrix_aligned(int n, int pad, float a[][maxN]) {
    float **align_A=new float*[n+pad];
    for (int i = 0; i < n; i++) {
       align_A[i] = (float*)malloc((n+pad) * sizeof(float));
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            align_A[i][j]=A[i][j];
        }
    }
    return align_A;
}

void gen_matrix(int n) {
	srand((unsigned)time(NULL));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = (float)rand() / RAND_MAX * rand_range;
			B[i][j] = (float)rand() / RAND_MAX * rand_range;
		}
	}
}

double timer(int count, void(*func)(int, float[][maxN], float[][maxN], float[][maxN]), int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	gen_matrix(n);
	(*func)(n, a, b, c);
	double sum = 0;
	for (; count > 0; count--) {
		gen_matrix(n);
		auto time_start = system_clock::now();
		(*func)(n, a, b, c);
		auto time_end = system_clock::now();
		auto duration = duration_cast<microseconds>(time_end - time_start);
		sum += duration.count();
	}
	return sum / us_per_ms;
}

void set_iter(int n, int* count, int* step) {
	if (n < 32) { *count = 20000; *step = 2; }
	else if (n < 64) { *count = 10000; *step = 4; }
	else if (n < 128) { *count = 2000; *step = 8; }
	else if (n < 256) { *count = 200; *step = 16; }
	else if (n < 512) { *count = 20; *step = 32; }
	else if (n < 1024) { *count = 20; *step = 64; }
	else if (n < 2048) { *count = 20; *step = 128; }
	else if (n < 4096) { *count = 20; *step = 256; }
}

int main()
{
	ofstream out("logs.txt", ios::app);
	int count;
	int step;
	double time;
	T = 64;
//	for (T = 16; T <= 256; T += 4 ) {
//		for (n = 16; n <= 1024; n += step) {
//			set_iter(n, &count, &step);
//			time = timer(count, sse_tile_norm, n, A, B, C);
//			cout << "sse_tile_norm" << '\t' << n << '\t' << T << '\t' << time << endl;
//			out << "sse_tile_norm" << '\t' << n << '\t' << T << '\t' << time / count << endl;
//		}
//	}
	for (n = 16; n <= 4096; n += step) {
		set_iter(n, &count, &step);
		time = timer(count, sse_mul, n, A, B, C);
		cout << "sse_mul" << '\t' << n << '\t' << T << '\t' << time << endl;
		out << "sse_mul" << '\t' << n << '\t' << T << '\t' << time / count << endl;
		//time = timer(count, trans_mul, n, A, B, C);
		//cout << "trans_mul" << '\t' << n << '\t' << T << '\t' << time << endl;
		//out << "trans_mul" << '\t' << n << '\t' << T << '\t' << time / count << endl;
		time = timer(count, serial_mul, n, A, B, C);
		cout << "serial_mul" << '\t' << n << '\t' << T << '\t' << time << endl;
		out << "serial_mul" << '\t' << n << '\t' << T << '\t' << time / count << endl;

		time = timer(count, sse_mul_openmp, n, A, B, C);
		cout << "sse_mul_openmp" << '\t' << n << '\t' << T << '\t' << time << endl;
		out << "sse_mul_openmp" << '\t' << n << '\t' << T << '\t' << time / count << endl;
		//time = timer(count, trans_mul_openmp, n, A, B, C);
		//cout << "trans_mul_openmp" << '\t' << n << '\t' << T << '\t' << time << endl;
		//out << "trans_mul_openmp" << '\t' << n << '\t' << T << '\t' << time / count << endl;
		time = timer(count, serial_mul_openmp, n, A, B, C);
		cout << "serial_mul_openmp" << '\t' << n << '\t' << T << '\t' << time << endl;
		out << "serial_mul_openmp" << '\t' << n << '\t' << T << '\t' << time / count << endl;
	}
	out.close();
	system("pause");
}


//void test()
//{
//      int a = 0;
//      for (int i=0;i<100000000;i++)
//          a++;
//}


//int main()
// {

//    clock_t t1 = clock();
//     for (int i=0;i<16;i++)
//         test();
//     clock_t t2 = clock();
//     std::cout<<"sequential time: "<<t2-t1<<std::endl;
//
//    clock_t t3 = clock();
//    #pragma omp parallel for
//     for (int i=0;i<16;i++)
//         test();
//     clock_t t4 = clock();
//     std::cout<<"parallel time: "<<t4-t3<<std::endl;
//
//
//    std::cout << "parallel begin:\n";
//
//    #pragma omp parallel
//
//    {
//
//        std::cout << omp_get_thread_num()<<std::endl;
//
//    }
//
//    std::cout << "\n parallel end.\n";
//
//     return 0;
// }
