#ifndef __CREATEIMG__
#define __CREATEIMG__

#include "main.h"

generic_images create_images(int n_images, int image_size, int init_val){

    // Given n_images, image_size and init_val
    // Returns a n_images generic_images, with size image_size
    // with init_val in each place

    // Create image
    generic_images images = _mm_malloc( n_images * sizeof(generic_image), 32);

    // For each image
    for(int img = 0; img < n_images; img++){

        // Create image
        generic_image image = _mm_malloc( image_size * sizeof(generic_row),32);

        // For each row
        for(int i = 0; i < image_size; i++){

            // Create row
            generic_row row = _mm_malloc( image_size * sizeof(float), 32);

            // For each column
            for(int j = 0; j < image_size; j++){

                // Writtes column value
                row[j] = init_val;
            }

            // Writtes row
            image[i] = row;
        }

        // Writtes image
        images[img] = image;
    }

    return images;
}

generic_images create_images_dif_val_row(int n_images, int image_size){

    // Given n_images, image_size
    // Returns a n_images generic_images, with size image_size
    // with value incresing by column, then by row, 
    // from 0 to height*weight for each image

    // Create image
    generic_images images = _mm_malloc( n_images * sizeof(generic_image), 32);
    int init_val = 0; 
    // For each image
    for(int img = 0; img < n_images; img++){

        // Create image
        generic_image image = _mm_malloc( image_size * sizeof(generic_row),  32);

        // For each row
        for(int i = 0; i < image_size; i++){

            // Create row
            generic_row row = _mm_malloc( image_size * sizeof(float), 32);

            // For each column
            for(int j = 0; j < image_size; j++){

                // Writtes column value
                row[j] = init_val;
                init_val++; 
            }

            // Writtes row
            image[i] = row; 
        }

        // Writtes image
        images[img] = image;
        init_val = 0;  
    }

    return images;
}

float* arange(int len){

    float* res = _mm_malloc(len * sizeof(float), 32);

    for(int i = 0; i < len; i++){

        res[i] = i;
    }

    return res;
}

#endif