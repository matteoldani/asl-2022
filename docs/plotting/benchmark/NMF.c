/*
*	NMF by alternative non-negative least squares using projected gradients
*	Original Author: Chih-Jen Lin, National Taiwan University
*	Original website: http://www.csie.ntu.edu.tw/~cjlin/nmf/index.html
*	C program auther: Dong Li (donggeat@gmail.com)
*	I rewrote the Matlab code in C, with BLAS. Free to use and modify.

*	As the first time to use BLAS and CBLAS, you may need to configure like this on Linux:
*	Download blas.tgz and cblas.tgz on http://www.netlib.org/blas/
*	1) Install BLAS, generate blas_LINUX.a
*	2) Modify the BLLIB in CBLAS/Makefile.in which link to blas_LINUX.a, and make all in CBLAS
*	3) Put the src/cblas.h to /usr/lib/ or somewhere your compiler can find it, then enjoy it!
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
#include <Accelerate/Accelerate.h>
#define bool int
#define false 0
#define true 1
#define max(a,b)(a>b)?a:b

//compile:	gcc NMF.c  -o NMF.o -c -O3 -DADD_ -std=c99
//		gfortran -o NMF_ex NMF.o /home/lid/Downloads/CBLAS/lib/cblas_LINUX.a /home/lid/Downloads/BLAS/blas_LINUX.a
//excute:	./NMF_ex

// gcc NMF.c  -o NMF.out -c -O3 -DADD_ -std=c99 -framework Accelerate
// gfortran -o NMF_ex NMF.out -framework Accelerate
// ./NMF_ex

void randomInit(double *data, int p)
{
	
    for (int i = 0; i < p; ++i)
	data[i] = rand() / (double)RAND_MAX;
}

//A:m*n,column-major; B=A'
void transpose(double *A, double *B, int m, int n){
	int k = 0;
	for( int i = 0; i < m; i++){
		for (int j = 0; j < n; j++)
			B[k++] = A[i+j*m];
	}		
} 

//W:m*n, H:n*k, V:m*k, grad:n*k
void nlssubprob(double *V, double *W, double *Hinit, int m, int n, int k, double tol, int maxiter, double *H, double *grad, int *ite){
	//H = Hinit; WtV = W'*V; WtW = W'*W; 
	memcpy(H, Hinit, n*k*sizeof(double));
	
	double *WtV = 0;		//WtV:n*k
	WtV = (double *)malloc(n*k*sizeof(double));
	memset(WtV, 0, n*k*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,k,m,1,W,m,V,m,0,WtV,n);
	
	double *WtW = 0;		//WtW:n*n
	WtW = (double *)malloc(n*n*sizeof(double));
	memset(WtW, 0, n*n*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,n,m,1,W,m,W,m,0,WtW,n);
	
	double alpha = 1; 
	double beta = 0.1;

	double *Hn = 0;	
	Hn = (double *)malloc(n*k*sizeof(double));
	memset(Hn, 0, n*k*sizeof(double));
	
	double *d = 0;
	d = (double *)malloc(n*k*sizeof(double));
	memset(d, 0, n*k*sizeof(double));
	
	double *WtWd = 0;
	WtWd = (double *)malloc(n*k*sizeof(double));
	memset(WtWd, 0, n*k*sizeof(double));
	
	double *Hp = 0;	
	Hp = (double *)malloc(n*k*sizeof(double));
	memset(Hp, 0, n*k*sizeof(double));
	
	double *Hnpp = 0;
	Hnpp = (double *)malloc(n*k*sizeof(double));
	memset(Hnpp, 0, n*k*sizeof(double));
	
	double *tmpvec = 0;		//WtW:n*n
	tmpvec = (double *)malloc(n*k*sizeof(double));

	int iter = 0;
	for ( iter = 1; iter <= maxiter; iter++){
		//grad = WtW*H - WtV;
		memcpy(grad,WtV,n*k*sizeof(double));
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,k,n,1,WtW,n,H,n,-1,grad,n);
		
		memset(tmpvec, 0, n*k*sizeof(double));
		int ii = 0;
		for (int i = 0; i < n*k; i++){
			if (grad[i] < 0 || H[i] > 0 )
				tmpvec[ii++] = grad[i];
		}
		double projgrad = cblas_dnrm2(ii, tmpvec, 1);
	
		if (projgrad < tol)
			break;
		bool decr_alpha = true;
		for (int inner_iter = 1; inner_iter <= 20; inner_iter++){
			//Hn = max(H - alpha*grad, 0); d = Hn-H;
			memcpy(Hn,H,n*k*sizeof(double));
			cblas_daxpy(n*k, -alpha, grad, 1, Hn, 1);
			for (int i = 0; i < n*k; i++){
				if (Hn[i] < 0)
					Hn[i] = 0;
			}
			 
			memcpy(d,Hn,n*k*sizeof(double));
			cblas_daxpy(n*k, -1, H, 1, d, 1);
			
			//gradd=sum(sum(grad.*d)); dQd = sum(sum((WtW*d).*d));
			double gradd = cblas_ddot(n*k, grad, 1, d, 1);
			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,k,n,1,WtW,n,d,n,0,WtWd,n);
			double dQd = cblas_ddot(n*k, WtWd, 1, d, 1);
			bool suff_decr = 0.99*gradd + 0.5*dQd < 0;
			//bool decr_alpha = true;
			if (inner_iter==1){
				decr_alpha = ~suff_decr; 
				memcpy(Hp,H,n*k*sizeof(double));
			}
			if(decr_alpha){
				if(suff_decr){
					memcpy(H,Hn,n*k*sizeof(double));
					break;
				}
				else
					alpha = alpha * beta;
			}
			else{
				memcpy(Hnpp,Hn,n*k*sizeof(double));
				cblas_daxpy(n*k, -1, Hp, 1, Hnpp, 1);
				if(~suff_decr || cblas_dnrm2(n*k,Hnpp,1) == 0){
					memcpy(H,Hp,n*k*sizeof(double));
					break;
				}
				else{
					alpha = alpha/beta;
					memcpy(Hp,Hn,n*k*sizeof(double));
				}
			}
		}
	}
	
	*ite = iter;
	if (*ite==maxiter)
		printf("Max iter in nlssubprob\n");
	
}

//NMF:V=W*H,W:m*n, H:n*k, V:m*k
//stick to BLAS, column-major
void NMF(double *V, double *Winit,double *Hinit, int m, int n, int k, double tol, double timelimit, int maxiter, double *W, double *H)
{
	//W = Winit; H = Hinit; initt = cputime;
	memcpy(W,Winit,m*n*sizeof(double));
	memcpy(H,Hinit,n*k*sizeof(double));
	clock_t initt = time(NULL);
	
	//gradW = W*(H*H') - V*H'; gradH = (W'*W)*H - W'*V;
	double *HHt = 0;		//grad:n*k
	HHt = (double *)malloc(n*n*sizeof(double));
	memset(HHt, 0, n*n*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n,n,k,1,H,n,H,n,0,HHt,n);
	
	double *gradW = 0;
	gradW = (double *)malloc(m*n*sizeof(double));
	memset(gradW, 0, m*n*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,n,1,W,m,HHt,n,0,gradW,m);
	
	double *VHt = 0;		//grad:n*k
	VHt = (double *)malloc(m*n*sizeof(double));
	memset(VHt, 0, m*n*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,m,n,k,1,V,m,H,n,0,VHt,m);
	cblas_daxpy(m*n, -1, VHt, 1, gradW, 1);
	
	double *WtW = 0;		//WtW:n*n
	WtW = (double *)malloc(n*n*sizeof(double));
	memset(WtW, 0, n*n*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,n,m,1,W,m,W,m,0,WtW,n);
	
	double *gradH = 0;
	gradH = (double *)malloc(n*k*sizeof(double));
	memset(gradH, 0, n*k*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,k,n,1,WtW,n,H,n,0,gradH,n);
	
	double *WtV = 0;
	WtV = (double *)malloc(n*k*sizeof(double));
	memset(WtV, 0, n*k*sizeof(double));
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,n,k,m,1,W,m,V,m,0,WtV,n);
	
	cblas_daxpy(n*k, -1, WtV, 1, gradH, 1);
	
	//double initgrad = norm([gradW; gradH'],'fro');
	double initgrad = cblas_ddot(m*n, gradW, 1, gradW, 1);
	initgrad += cblas_ddot(n*k, gradH, 1, gradH, 1);
	initgrad = sqrt(initgrad);
	double tolW = initgrad*max(0.001,tol); 
	double tolH = tolW;
	
	double *tmpvec = 0;		//WtW:n*n
	tmpvec = (double *)malloc(m*n*sizeof(double));
	memset(tmpvec, 0, m*n*sizeof(double));
	
	double *tmpvec2 = 0;		//WtW:n*n
	tmpvec2 = (double *)malloc(n*k*sizeof(double));
	memset(tmpvec2, 0, n*k*sizeof(double));
	
	double *Vt = 0;		
	Vt = (double *)malloc(m*k*sizeof(double));
	memset(Vt, 0, m*k*sizeof(double));
	
	double *Ht = 0;		
	Ht = (double *)malloc(n*k*sizeof(double));
	memset(Ht, 0, n*k*sizeof(double));
	
	double *Wt = 0;		
	Wt = (double *)malloc(m*n*sizeof(double));
	memset(Wt, 0, m*n*sizeof(double));
	
	double projnorm = 0;
	int iter = 0;	
	for (iter = 1; iter <= maxiter; iter++){
		//projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);

		int ii = 0;
		for (int i = 0; i < m*n; i++){
			if (gradW[i] < 0 || W[i] > 0 )
				tmpvec[ii++] = gradW[i];
		}
		projnorm = cblas_ddot(ii, tmpvec, 1, tmpvec, 1);
		ii = 0;
		for (int i = 0; i < n*k; i++){
			if (gradH[i] < 0 || H[i] > 0 )
				tmpvec2[ii++] = gradH[i];
		}
		projnorm += cblas_ddot(ii, tmpvec2, 1, tmpvec2, 1);
		projnorm = sqrt(projnorm);
		
		if ( projnorm < tol*initgrad || time(NULL) - initt > timelimit )
			break;
		
		//[W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000);
		int iterW = 0;
		
		transpose(V, Vt, k, m);
		transpose(H, Ht, n, k);
		transpose(W, Wt, m, n);
		nlssubprob(Vt, Ht, Wt, k, n, m, tolW, 1000, W, gradW, &iterW);//printf("complete stage %d\n",iter);
		transpose(W, W, m, n);
		transpose(gradW, gradW, m, n);

		if (iterW == 1)
			tolW = 0.1 * tolW;
		
		int iterH = 0;
		nlssubprob(V, W, H, m, n, k, tolH, 1000, H, gradH, &iterH);//printf("complete stage %d\n",iter);
		if (iterH == 1)
			tolH = 0.1 * tolH;
		
		if ( iter%10 == 0)
			printf(".");
		//printf("\nIter = %d tol=%f proj-grad norm %f\n", iter, tol, projnorm);
	}
	
}

int main(int argc, char **argv)
{	
	// An easy example to show the usage of NMF function
	int input_sizes[5] = {45000, 50000, 55000, 60000, 65000};
	for(int i = 0; i < 5; i++) {
		srand((unsigned)time(NULL));
		int m = input_sizes[i];
		int n = 3;
		int k = m;
		
		//double V[8]={1,2,3,4,5,6,7,8};
		double *V = 0;		
		V = (double *)malloc(m*k*sizeof(double));
		randomInit(V, m*k);

		double *Winit = 0;		
		Winit = (double *)malloc(m*n*sizeof(double));

		double *W= 0;		
		W = (double *)malloc(m*n*sizeof(double));
		memset(W, 0, m*n*sizeof(double));

		double *Hinit = 0;		
		Hinit = (double *)malloc(n*k*sizeof(double));
		double *H= 0;		
		H = (double *)malloc(n*k*sizeof(double));
		memset(H, 0, n*k*sizeof(double));

		randomInit(Winit, m*n);
		randomInit(Hinit, n*k);

	    clock_t start, end;
        start = clock();
		NMF(V, Winit, Hinit, m, n, k, 0.0001, 9999999999999999, 1000, W, H);
        end = clock();

		printf("%i\n", m);
		printf("%f\n", ((end - start) / (double) CLOCKS_PER_SEC));
		
		free(Hinit);
		free(Winit);
		free(H);
		free(W);
	}
	exit(1);
}
