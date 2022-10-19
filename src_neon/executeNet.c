#ifndef __EXECUTENET__
#define __EXECUTENET__

#include "main.h"

float* executeNet(net* net, generic_images imgs){

    float* vect;

    for(int j = 0; j < (net->cant_capas); j++){

        switch(net->arq_names[j]){

            case 'C':
                imgs = conv(imgs, net->arq_structs[j]); 
                break;

            case 'M':
                imgs = maxpool(imgs,  net->arq_structs[j]); 
                break;

            case 'F':
                vect = flatten(imgs, net->arq_structs[j]);
                break;

            case 'D':

                vect = dense(vect, net->arq_structs[j]);
                break;

            default:

                printf("Error to identify layer %u",  j); 
                exit(-1); 
                break;

        }

    } 

    // recoleccion de metricas y tiempos

    // almacen de prediccion

    return vect;
}

#endif
