#ifndef __LOADIMAGE__
#define __LOADIMAGE__

#include "main.h"


void loadData(unsigned int* loaded_labels, dataset imgs){
 
    FILE* labels = fopen("../data/t10k-labels-idx1-ubyte", "rb"); 
    FILE* images = fopen("../data/t10k-images-idx3-ubyte", "rb"); 
    
    if(labels == NULL){
        printf("Error al cargar los labels");  
        exit(-1); 
    }
    if(images== NULL){
        printf("Error al cargar las imagenes");  
        exit(-1); 
    } 
    //printf("Labels y Images cargadas \n"); 
    
    //labels 
    unsigned char magic_number1[4];
    fread(magic_number1, 4, 1 ,labels); 

    unsigned char num_items[4];  
    fread(num_items, 4, 1, labels); 
    
    //images 
    unsigned char magic_number2[4];
    fread(magic_number2, 4, 1 ,images); 

    unsigned char numb_images[4]; 
    fread(numb_images, 4, 1, images); 

    unsigned char nrow[4]; 
    unsigned char ncol[4]; 
    fread(nrow, 4, 1, images); 
    fread(ncol, 4, 1 , images);
    
    //images
    for (int i = 0; i < test_size; i++)
    { //for each image 
        unsigned char label; 
        fread(&label,  1, 1, labels);
         
        loaded_labels[i] = label;  

        generic_image im = _mm_malloc(sizeof(generic_row) * height,32); 
        for (int j = 0; j < height; j++)
        {
            generic_row row = _mm_malloc(sizeof(float) * width, 32); 
            for (int k = 0; k < width; k++)
            {
                unsigned char pixel; 
                fread(&pixel,  1, 1, images);  
                row[k] = (float) pixel/255;     
            }
            im[j] = row; 
        }
        generic_images ims = _mm_malloc(sizeof(generic_image), 32); 
        ims[0] = im; 
        imgs[i] = ims;  
    }
    
}
#endif