/*
  This program solves the NLS equation with the split-step method
  including the Raman response.
  Ref.: Skupin et al. PR E 70, 046602 (04)

  spatial routine:FFT method convolution (ooura FFT)
  nonlinear routine: exponential multiplication
                    (self focusing+Plasma defocusing+multiphoton absorption)
   */
/* z adjustable step, loop exit and save for collapse*/

   
/* comilation with gsl
g++ -w -Ofast -I/usr/local/include/ -c LGspacelin.cpp -lfftw3 -fopenmp

g++ -L/usr/local/lib/ LGspacelin.o -lgsl -lgslcblas -lm -lfftw3 -fopenmp

./a.out
   
  Property of 
  Femtosecond spectroscopy Lab.
   Binghamton Univeristy    2020
 */

// MKS unit

#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_gamma.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <dos.h>
#include <math.h>
//#include <conio.h>
#include <complex>
#include "nr.h"
#include "nrutil.h"
#include "nrutil.cpp"
#include "fft4g.c"
#include "fft4g2d.c"
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define pi 3.141592
#define NMAX 8192
#define NMAXSQRT 64
#define nz 10000 //maximum step

typedef complex<double> Complex;
Complex I_UNIT(0.,1.);

//input parameter specification
   double w_x = 100.0e-6, // spot size (in x radius)
          w_y =100.0e-6,//  spot size (in y radius)
          w_0=sqrt(w_x*w_y),// average spotsize
                            
          c_MKS=2.99792458e8,  // speed of light MKS
          c_CGS=2.99792458e10,  // speed of light CGS
          h_bar=6.626e-34/(2.*pi), // planck constant J.s
          e_MKS = 1.602e-19,  // Fundamental charge, in Coulomb
          me_MKS = 9.109e-31,  // Electron mass, in kg
          epsilon=8.85e-12, //permittivity of free space (F/m)
          mu=4.*pi*1.e-7, // Permeability of free space (H/m)
          n_0=1.0, //medium index
          fac_intsy=epsilon*c_MKS*n_0/2., //multiplication factor for intensity 

          lambda=8.e-7, //laser wavelength in m
          omega=2.*pi/lambda*c_MKS,// frequency of the central wavelength
          k0 = 2.0*pi/lambda,//radians/meter central wavenumber 
          n_2=2.0e-20, //nonlinear index (m^2/W)
          w_min=w_0,// minimum spot size at the laser focus
          z_r=n_0*pi*w_min*w_min/lambda,// Rayleigh range for smallest beam
          zInitial = 1.0e-100,//Initial distance from the laser focus to the start of the beam in m  if beam is collimated tiny number
          radCurve = pow(z_r,2.0)/zInitial + zInitial,// initial radius of curvature of the beam in m
	  zdistance =2.e-2,// m
          g=1.8962,//2.0267, critical power factor gaussian:1.8962 SG;2.0267
          P_c=g*(lambda*lambda)/(4.*pi*n_0*n_2), //critical power
          P=5.*P_c,//power in Watts
          BETA_n=4.25e-48,//Multiphoton ionization contant MKS ************** Should be calculated by us  
          //bg1=k0*n2de_alpha*alpha*L_df/L_nl,//
         // bg2=L_df/L_pl,//
         // bg3=L_df/L_mp/2./sqrt(n_mp),//
         
          noi=0.1;//noise magnitude percent   
        
    
    //this is the parameter for the order of the OAM mode 
    int nLag = 0;// radial index
    int  aLag = 0;//azimuthal index (number of charges)
   
   
    
//function declartion
void cdft2d(int, int, int, double **, double *,int *, double *);//for FFT
void to_real2d(int , int , Complex **, double **);//Complex to real data
void to_comp2d(int , int , Complex **, double **);//real to Complex data
void IC (Complex **, int, int, double *, double *);  //initial condition
void cal_power(Complex **, int, int, double , double, double );//power calculation
void fileoutxy(char *, double *, double *, int);//xy fileout
void fileout(char *, double, double **, int , int ); //File output function

 int main() {

   clock_t start, end;
   start = clock();

   double beta;//nonlinear factor
   int j, k, l, l2, l3, p, q;
   /*FFT preparation */
   int ip[NMAXSQRT + 2]; //for FFT
   static double ufft__[NMAX][NMAX], *ufft[NMAX], t[2 * NMAX],
        w[NMAX * 3 / 2];
    for (j = 0; j < NMAX; j++) ufft[j] = ufft__[j];


   //int nx=2048,ny=2048;   // powers of 2 for FFT
  int nx=1024,ny=1024; 

  ip[0]=0; //first time use

  double dz, zstp0=1.e-4, zpar=1./zstp0, // step, intial step and adaptive step factor in m
         dist, dx= 1.e-6, dy= 1.e-6;//spatial resolution in m


  int fsav=20; // file save number
  double back_step=zdistance/(double)fsav;//file save zdistance from the last z value
  char f_namei[20];
 // int n_bs=(int)(zdistance/back_step);

  dist=0.;


   //spatial frequencies and z arrays
   double *x, *y, *z, *f_x, *f_y;
   double fs_x,fs_y; //sampling frequency simply (1/dx and 1/dy) k= 2pi *sampling frequency
   x=dvector(0,nx-1);
   y=dvector(0,ny-1);
   z=dvector(0,nz-1);
   f_x=dvector(0,nx-1);
   f_y=dvector(0,ny-1);

   //assignment
   for (j=0;j<nx;j++)
   x[j]=(double)(j-(nx/2))*dx;
   for (j=0;j<ny;j++)
   y[j]=(double)(j-(ny/2))*dy;
   for (j=0;j<nz;j++)
   z[j]=(double)(j+1)*dz;


   fs_x=1./dx; fs_y=1./dy ;

    for (j=0;j<nx;j++)
     f_x[j]=(j<=nx/2)? fs_x/nx*((double)(j)):fs_x/nx*((double)(j-nx));
    for (j=1;j<ny;j++)
     f_y[j]=(j<=ny/2)? fs_y/ny*((double)(j)):fs_y/ny*((double)(j-ny)) ;



   //field assignment the independent variable is radius
  Complex **u,**U; //u: electric field in real space U: electric field in k space 
  double **E, **I,**I_f ; //E: real electric field, I: intensity
  double Imax,Imax1,IExit, *I_max; //maximum intensity at each step
  double E_i, E_out; // Energy (power) conservation check

  u=compmatrix(0,nx-1,0,ny-1);
  U=compmatrix(0,nx-1,0,ny-1);
  E=dmatrix(0,nx-1,0,ny-1);
  I=dmatrix(0,nx-1,0,ny-1);
  I_f=dmatrix(0,nx-1,0,ny-1);
  I_max=dvector(0,nz-1);

   // Initial condition assianment
 // beta=bg1;//initial value
  IC(u,nx,ny,x,y);
  
  cal_power(u,nx,ny,dx,dy,beta); // This should be equla to the input power without noise
  //getchar(); 

  //intensity and energy calculation
   E_i=0;
  for (j=0;j<nx;j++) {
     for (k=0;k<ny;k++) {
          E[j][k]= real(u[j][k]); //field profile only for input beams to check the charge
          I[j][k]= abs(u[j][k])*abs(u[j][k]);
      }
   }
 
 //finding maximum intensity

     Imax1=0.; //initialize
     for (p=0;p<nx;p++) {
        for (q=0;q<ny;q++) {
            Imax1=max(Imax1,I[p][q]);
        }
     }
     
  IExit=1000.*Imax1;

   for (j=0;j<nx;j++) {
     for (k=0;k<ny;k++)
      {E_i+=I[j][k]*dx*dy;}
  }


  double L_nl=1./(2.*pi/lambda*n_2*Imax1);


  //initial field and intensity fileout
  fileout ("Intensity z=0",dist,I,nx,ny);
  fileout ("Electric field z=0",dist,E,nx,ny); 
 
   // Now the laser propagation!,operator splitting
  printf("Input power=%g W\n",P); 
  printf("critical power=%g W\n",P_c);
  printf("peak intensity=%g W/m^2\n",Imax1);
  printf("Input power in unit of P_cr=%g\n",P/P_c);
  printf("propagation zdistance=%g cm\n",zdistance*1.e2);
  printf("Nonlinear length=%g cm\n",L_nl*1.e2);
  if (L_nl>zdistance) printf("Nonlinear length is longer than propagation distance. Increase the distance!\n"); 
  getchar();
  printf("dx=%g um\n",dx*1.e6);
  printf("dy=%g um\n",dy*1.e6);
  printf("initial z-step size=%g um \n",zstp0*1.e6);
  printf("file saved step backward=%g cm\n\n",back_step*1.e2);
 
  dz=zstp0; //initial step assignment
  l2=0;//z-store step index
  l3=fsav;

  for (l=0;l<nz ;l++ )
  {


   dist+=dz;
   z[l]=dist;
   printf("z=%g cm\n",dist*1.e2);

   to_real2d(nx,ny,u,ufft); //preparation for FFT 
   //Nonlinear first half-step:simple multiplication in temporal domain

    for (j=0;j<nx;j++) {
       for (k=0;k<ny;k++) {
         u[j][k]= exp(I_UNIT*k0*n_2*I[j][k]*dz/2.)*u[j][k];
         I[j][k]=abs(u[j][k])*abs(u[j][k]);

        }
     }
   
   
   // Diffraction: convolution calculation using 2D FFT
  //  to_real2d(nx,ny,u,ufft); //preparation for FFT
    cdft2d(nx, 2*ny, 1, ufft, t, ip, w); // FFT

    to_comp2d(nx,ny,U,ufft);


    for (j=0;j<nx;j++) {//Multiplication of propagation term
        for (k=0;k<ny;k++) {
          U[j][k]=U[j][k]*exp(-I_UNIT*(pi*lambda)*(f_x[j]*f_x[j]+f_y[k]*f_y[k])*dz);
        }
     }
     to_real2d(nx,ny,U,ufft);

     cdft2d(nx, 2*ny, -1, ufft, t, ip, w);  // inverse FFT

     to_comp2d(nx,ny,u,ufft);

    for (j=0;j<nx;j++) {
       for (k=0;k<ny;k++) {
          u[j][k]=u[j][k]/(double)(nx*ny); //normalizatione
          I[j][k]=abs(u[j][k])*abs(u[j][k]);
       }
   }

   //Nonlinear second half-step:simple multiplication in temporal domain
   for (j=0;j<nx;j++) {
       for (k=0;k<ny;k++) {
         u[j][k]= exp(I_UNIT*k0*n_2*I[j][k]*dz/2.)*u[j][k];
         I[j][k]=abs(u[j][k])*abs(u[j][k]);

        }
     }

    //finding maximum intensity

     Imax=0.; //initialize
     for (p=0;p<nx;p++) {
        for (q=0;q<ny;q++) {
            Imax=max(Imax,I[p][q]);
        }
     }
     I_max[l2]=Imax;  //maximum intensity
     l2=l2+1;//step forward
     printf("maximum intensity is =%g W/m^2\n",Imax);

    if (Imax>IExit) break; // if it collapses, exit from the loop

   //intermediate step fileoutput

    if (fabs(dist-(zdistance-double(l3)*back_step))<2*dz && l3>=1) {
        sprintf(f_namei,"Intensity%d",fsav-l3+1);
        fileout (f_namei,dist,I,nx,ny);
        l3--;
        printf("file saved at z=%g cm\n",dist*1.e2);
        printf("%i more files will be saved\n\n",l3); }

   //Now the step adjust
     dz=1./((1./zstp0)+ zpar*(sqrt(Imax/Imax1)-1.));
    if ((dist+dz-zdistance)*(dist+dz-0.)>0.){
        dz=(zdistance-dist); //if overshoot, decrease
        l=nz-1; //then, the next step is the last step
    }

 } //end of the main Loop



  //Energy calculation
    E_out=0.;
    for (j=0;j<nx;j++) {
     for (k=0;k<ny;k++)
      {E_out+=I[j][k]*dx*dy;}
   }

    printf("initial energy=%g\n",E_i);
    printf("final energy=%g\n",E_out);




    //Final pulse output
    fileout ("Intensity final",dist,I,nx,ny);

    //distnace vs peak intensity
    fileoutxy("peak intensity vs zdistance",z,I_max,l2);

 //memory clean-up


  free_dvector(I_max,0,nz-1);
  free_dmatrix(I_f,0,nx-1,0,ny-1);
  free_dmatrix(I,0,nx-1,0,ny-1);
  free_compmatrix(U,0,nx-1,0,ny-1);
  free_compmatrix(u,0,nx-1,0,ny-1);

  free_dvector(f_y,0,ny-1);
  free_dvector(f_x,0,nx-1);
  free_dvector(z,0,nz-1);
  free_dvector(x,0,nx-1);
  free_dvector(y,0,ny-1);

  end = clock();
  printf("The time was: %f\n", (end - start));

  //getchar();
  return 0;
 }

 void to_real2d(int n1, int n2, Complex **a, double **b) {

 int j,k;
 for (j=0;j<n1;j++) {
     for (k=0;k<n2;k++) {
          b[j][2*k]= real(a[j][k]);
          b[j][2*k+1]=imag(a[j][k]);
      }
   }
 }

 void to_comp2d(int n1, int n2, Complex **a, double **b) {

 int j,k;
 for (j=0;j<n1;j++) {
     for (k=0;k<n2;k++) {
          a[j][k]=b[j][2*k]+I_UNIT*b[j][2*k+1];
      }
   }
 }

    /* Initial condition including the noise*/

void IC (Complex **a, int n_x, int n_y,  double *xx, double *yy){

void fileout (char *, double , double **, int , int );

double prefactor = sqrt(2.0*gsl_sf_fact(nLag)/pi/gsl_sf_fact((int)aLag + nLag));

 int j,k;
 long seed1=-3, seed2=-4;

 float r1, r2;
 double theta, **phase;
 
 phase=dmatrix(0,n_x-1,0,n_y-1);


 for (j=0;j<n_x;j++){
    for (k=0;k<n_y;k++) {
              r1=ran1(&seed1), r2=ran1(&seed2);

              double r = sqrt(pow(xx[j],2.0) + pow(yy[k],2.0));
               
               if (xx[j]==0.)theta=(yy[k]>0.)? pi/2.:-pi/2.;
               else theta=atan2(yy[k],xx[j]);
              
               a[k][j] =sqrt(P)*prefactor*(1/w_0)*exp(-I_UNIT*k0*pow(r,2.0)/2.0/ radCurve -pow(r/w_0,2.0) - I_UNIT*(double)(aLag)*theta)* 
                        pow( (r*sqrt(2.0)/w_0),aLag)*gsl_sf_laguerre_n(nLag,aLag,2.0*pow(r/w_0,2.0))*(1.+noi*exp(I_UNIT*(double)r2*2.*pi));
             
               phase[k][j]=-(double)(aLag)*theta;              
     }
  }
             
 //Initial phase 
  fileout ("Input beam phase", 0. , phase,n_x,n_y);
  free_dmatrix(phase,0,n_x-1,0,n_y-1);
}

//power calculation
void cal_power(Complex **a, int n_x, int n_y, double x_st, double y_st, double nonl) {

  int j,k;
  double power;

  power=0.;
  for (j=0;j<n_x;j++){
    for (k=0;k<n_y;k++) {
        power=power+(real(a[j][k])*real(a[j][k])+imag(a[j][k])*imag(a[j][k]))*x_st*y_st; //power calculation;

      }
  }
 // nonl=nonl*1.8962*(pi/2.)/power;
  printf("Calculated input power=%g W\n",power);
}



 float ran1(long *idum) {
  int j;
  long k;
  static long iy=0;
  static long iv[NTAB];
  double temp;
  if (*idum <= 0 || !iy) { //Initialize.
      if (-(*idum) < 1) *idum=1; //Be sure to prevent idum = 0.
      else *idum = -(*idum);
      for (j=NTAB+7;j>=0;j--) {// Load the shue table (after 8 warm-ups).
           k=(*idum)/IQ;
           *idum=IA*(*idum-k*IQ)-IR*k;
           if (*idum < 0) *idum += IM;
           if (j < NTAB) iv[j] = *idum;
      }
     iy=iv[0];
  }
  k=(*idum)/IQ; //Start here when not initializing.
  *idum=IA*(*idum-k*IQ)-IR*k; //Compute idum=(IA*idum) % IM without overif
  if (*idum < 0) *idum += IM; //flows by Schrage's method.
  j=iy/NDIV; //Will be in the range 0..NTAB-1.
  iy=iv[j];//output previously stored value and rell hte
  iv[j] = *idum; //shue table.
  if ((temp=AM*iy) > RNMX) return RNMX; //Because users don't expect endpoint values.
  else return temp;
 }




  //File output function (x,y)
void fileoutxy(char *flnm, double *x, double *y, int n) {

    int j;
    FILE *outfile;
   outfile=fopen(flnm,"w");
   if (outfile==NULL) {
    printf("Couldn't open %s for output.\n",flnm);
   } else {
        for (j=0;j<n;j++) fprintf(outfile,"%g %g\n",x[j],y[j]);
      fclose(outfile);
   }
}

//File output function
void fileout (char *filenm, double d, double **a, int n_x, int n_y ) {
    int j,k;
    FILE *outfile;
   outfile=fopen(filenm,"w");
   if (outfile==NULL) {
    printf("Couldn't open %s for output.\n",filenm);
   } else {
      fprintf(outfile,"%g\n",d);
      for (j=0;j<n_x;j++) {
         for (k=0;k<n_y;k++) {
           fprintf(outfile,"%g\t",a[j][k]);
         }
      fprintf(outfile,"\n");
     }
        fclose (outfile);
   }
 }
