
#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif

#include <stdlib.h>

double wtime()
{
#ifdef _OPENMP
   /* Use omp_get_wtime() if we can */
   return omp_get_wtime();
#else
   /* Use a generic timer */
   static int sec = -1;
   struct timeval tv;
   gettimeofday(&tv, NULL);
   if (sec < 0) sec = tv.tv_sec;
   return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
#endif
}

    
