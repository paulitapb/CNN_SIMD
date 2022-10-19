#ifndef __MEASURE_TIME__
#define __MEASURE_TIME__

#include "main.h"
//measure with rdtsc. requires intrinsics lib
inline
uint64_t readTSC() {
    _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    uint64_t tsc = __rdtsc();
    _mm_lfence();  // optionally block later instructions until rdtsc retires
    return tsc;
}


void measure_rdtsc_all_layers(net* loaded_net, generic_images imgs){ //works
    //measure clock cycles with rdtsc using intrinsics
    ticks tick,tick1;
    double time =0;

    tick =readTSC(); //start measurement

    //function to measure
    float* res = executeNet(loaded_net, imgs);
    
    tick1 = readTSC(); //stop measurement

    printf("=rdtsc= ticks: %u \n",tick1-tick); 

    time = (float)(((tick1-tick)+1500000)/2300000); // 2.3GHz CPU 
    //printf("=rdtsc= time in ms: %lf \n",time);
    //printf("=rdtsc= time in s: %lf \n",time/1000.0);
}

void measure_rdtsc_by_layer(net* loaded_net, generic_images imgs){ //works
    //measure clock cycles with rdtsc using intrinsics
    ticks tick,tick1,t1,t2;
    double time =0;
    printf("no SIMD\n");  
    t1 = readTSC(); 
    //function to measure
    float* vect; 
    for(int j = 0; j < (loaded_net->cant_capas); j++){

        switch(loaded_net->arq_names[j]){

            case 'C': 
                tick =readTSC(); //start measurement
                imgs = conv(imgs, loaded_net->arq_structs[j]);
                tick1 = readTSC(); //stop measurement
                printf("conv ticks: %u \n",tick1-tick); 
                break;

            case 'M':
                tick =readTSC(); //start measurement
                imgs = maxpool(imgs,  loaded_net->arq_structs[j]);
                tick1 = readTSC(); //stop measurement
                printf("maxp ticks: %u \n",tick1-tick); 
                break;

            case 'F':
                tick =readTSC(); //start measurement
                vect = flatten(imgs, loaded_net->arq_structs[j]);
                tick1 = readTSC(); //stop measurement
                printf("flatten ticks: %u \n",tick1-tick);
                break;

            case 'D':
                tick =readTSC(); //start measurement
                vect = dense(vect, loaded_net->arq_structs[j]);
                tick1 = readTSC(); //stop measurement
                printf("dense ticks: %u \n",tick1-tick); 
                break;

            default:

                printf("Error to identify layer %u",  j); 
                exit(-1); 
                break;

        }

    }
    
    t2 = readTSC(); //stop measurement

    printf("=rdtsc= ticks: %u \n",t2-t1); 
}

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
//requires time.h
void measure_gettimeofday_linux(net* loaded_net, generic_images imgs){
    struct timeval t1, t2;
    double elapsedTime;

    // start timer
    gettimeofday(&t1, NULL);

    // do something
    float* res = executeNet(loaded_net, imgs);

    // stop timer
    gettimeofday(&t2, NULL);

    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("=gettimeofday= %f ms.\n", elapsedTime); 
    printf("=gettimeofday= %f s.\n", elapsedTime/1000.0);
}
void measure_gettimeprocess_linux(net* loaded_net, generic_images imgs){
    struct timeval t1, t2;
    double elapsedTime;

    // start timer
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

    // do something
    float* res = executeNet(loaded_net, imgs);

    // stop timer
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("=gettimeprocess= %lf ms.\n", elapsedTime); 
    printf("=gettimeprocess= %lf s.\n", elapsedTime/1000.0); 
}

void measure_gettimerealtime_linux(net* loaded_net, generic_images imgs){
    struct timeval t1, t2;
    double elapsedTime;

    // start timer
    clock_gettime(CLOCK_REALTIME, &t1);

    // do something
    float* res = executeNet(loaded_net, imgs);

    // stop timer
    clock_gettime(CLOCK_REALTIME, &t2);

    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("=getrealtime= %lf ms.\n", elapsedTime); 
    printf("=getrealtime= %lf s.\n", elapsedTime/1000.0); 
}

#endif