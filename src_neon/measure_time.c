#ifndef __MEASURE_TIME__
#define __MEASURE_TIME__

#include "main.h"

//requires time.h
void measure_time_clock(net* loaded_net, generic_images imgs){
    clock_t start, end; 
    double elapsed_time; 
    
    start = clock(); 

    //function to measure
    float* res = executeNet(loaded_net, imgs);

    end = clock(); 

    elapsed_time = ((double)(end-start))/CLOCKS_PER_SEC;
    
    printf("=clock= time in sec %f \n", elapsed_time); 
}

#endif
