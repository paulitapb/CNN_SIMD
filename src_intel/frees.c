#ifndef __FREES__
#define __FREES__

#include "main.h"

void free_generic_image(generic_image img, int n_generic_row){

    for(int i = 0; i < n_generic_row; i++){

        _mm_free(img[i]);
    }

    _mm_free(img);
}

void free_generic_images(generic_images imgs, int n_generic_image, int n_generic_row){

    for(int i = 0; i < n_generic_image; i++){

        free_generic_image(imgs[i], n_generic_row);
    }

    _mm_free(imgs);
}

void free_data_set(dataset datas, int n_used, int n_generic_images, int n_generic_image, int n_generic_row){

    for(int i = n_used; i < n_generic_images; i++){

        free_generic_images(datas[i], n_generic_image, n_generic_row);

    }

    _mm_free(datas);    

}

void free_conv_layer(conv_struct* conv_layer){

    int n_img_in = conv_layer->dim_in[0];
    int n_img_out = conv_layer->dim_out[0];

    int kernel_size = conv_layer->kernel_size;

    for(int i = 0; i < n_img_in; i++){

        for(int j = 0; j < n_img_out; j++){

            for(int k = 0; k < kernel_size; k++){

                _mm_free(conv_layer->kernels[i][j][k]);
            }

            _mm_free(conv_layer->kernels[i][j]);
        }

        _mm_free(conv_layer->kernels[i]);
    }

    _mm_free(conv_layer->kernels);

    _mm_free(conv_layer->bias);

    free(conv_layer);
}


void free_dense_layer(dense_struct* dense_layer){
     
    for(int i = 0; i < dense_layer->dim_out; i++){

        _mm_free(dense_layer->weights[i]);
    }
    
    _mm_free(dense_layer->weights);

    _mm_free(dense_layer->bias); //ROMPE TODO ESTO
    
    free(dense_layer);

}

void free_net(net* net_free){

    for(int i = 0; i < net_free->cant_capas; i++){

        char layer_name = net_free->arq_names[i];

        switch(layer_name){ // C: convolution, M: maxpool, F: flatten, D: dense

            case 'C': // convolution layer case

                free_conv_layer(net_free->arq_structs[i]);
                break;

            case 'M': // maxpooling layer case

                free(net_free->arq_structs[i]);
                break;

            case 'F': // flatten layer case

                free(net_free->arq_structs[i]);
                break;

            case 'D': // dense layer case

                free_dense_layer(net_free->arq_structs[i]);
                break;

            default: // not a recognazible layer

                printf("Error al liberar memoria de Red\n"); 
                break; 
        }

    }

    free(net_free->arq_names);
    free(net_free->arq_structs);
    free(net_free);
}

#endif