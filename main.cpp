#include <iostream>
#include <random>

using namespace std;


void compute_cost_F(double*& F, const double* x, const double* x_ref, const int& N)
{
    double temp;
    for(int i = 0; i < N; i++)
    {
        temp = x_ref[i] - x[i];
        F[i] = temp * temp;
    }
    return;
}


int main(int argc, char* argv[])
{
    double* x_ref;
    double* x;
    double* coord;
    double* F;
    doubl eps;

    eps = 1e-4; // an error tolerance;
    N = 100;
    x_ref = new double[N];
    x     = new double[N];
    coord = new double[N];
    F     = new double[N];

    double d_coord = 2./(N-1);
    for(int i = 0; i < N; i++)
        coord[i] = -1. + d_coord*i;

    // reference signal:
    for(int i = 0; i < N; i++)
        x_ref[i] = coord[i]*coord[i];

    // initial guess:
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    for(int i = 0; i < N; i++)
        x[i] = unif(gen);

    // initial cost function:
    compute_cost_F(F, x, x_ref, N);

    /**
     * > find derivative of the cost function to compute g_k = partial_x F(x) (compact finite difference or standart finite difference);
     * > find an initial symmetric positive-definite Hessian;
     * > if ||g|| < eps (choose between norm-inf, norm-l1, norm-l2), stop -> take y;
     * > otherwise, two-loop recursion for H_k g_k to compute -d_k;
     * > use the backtracking line search to find alpha_k;
     * > set x_{k+1} = x_k + alpha_k d_k
     * > set s_k = x_{k+1} - x_k
     * > set y_k = g_{k+1} - g_k
     * > store the pair (s_k, y_k)
     * > take next guess for the Hessian H_k^0 = s_k^T y_k/||y_k||^2 I
    */


    return 0;
}