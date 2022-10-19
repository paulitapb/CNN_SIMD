#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// intrinsics  
#include <immintrin.h> // AVX
#include <x86intrin.h> //all instrinsics 

// Defines
#define test_size 10000
#define height 28
#define width 28

// ---- Declaraciones de structs ----

// Declaraciones de structs

    // struct generic image  - aligned

typedef float* generic_row;

typedef generic_row* generic_image;

typedef generic_image* generic_images;

typedef generic_images* dataset;

typedef struct data
{
    //not aligned
    unsigned int label; 
    generic_image img; 

} data;

// struct network
typedef struct cnn_net
{
    //not aligned
    int cant_capas; 
    unsigned char* arq_names; //layer_ids
    void** arq_structs; //layer_structs 
} net;

// struct for convolutional kernels

    // conv kernels - aligned
typedef float* conv_kernel_row;
typedef conv_kernel_row* conv_kernel;
typedef conv_kernel* conv_kernels_row;
typedef conv_kernels_row* conv_kernels;

    //conv bias  - aligned
typedef float* conv_bias;

    // struct convolution
typedef struct conv_struct{

    int dim_in [2]; // dimension input image
    int dim_out [2]; // dimension output image
    int kernel_size; // dimension del kernel
    conv_kernels kernels; // pointer to kernels matrix of matrixes(defined in run time) - aligned
    conv_bias bias; // pointer to bias array (defined in run time) -aligned
    int strides; // strides

} conv_struct;

// struct maxpool
typedef struct maxpool_struct
{ 
    int dim_in[2]; 
    int dim_out[2];  
    int pool_size;
    int strides; 
}maxpool_struct;

// struct flatten
typedef struct flatten_struct{
    int dim_in[3]; // dim_in[0] = cantImages, dim_in[1] = height, dim_in[2] = width
    int dim_out;
}flatten_struct; 

// struct dense
typedef conv_kernel_row dense_kernel_row;
typedef conv_kernel dense_kernel;
typedef float* dense_bias; //aligned

typedef struct dense_struct
{
    int dim_in;
    int dim_out;
    unsigned char actFunc[4];  
    dense_kernel weights; // aligned 
    dense_bias bias; //aligned
}dense_struct;


// ---- Declaraciones de funciones ----

// loadImage.c
void loadData(unsigned int* loaded_labels, dataset imgs); 

// loadNet.c

// read data types: export for test only
int read_int(FILE *file_ptr, long int *file_idx);
float read_float(FILE *file_ptr, long int *file_idx);
char* read_string(FILE *file_ptr, long int *file_idx);

// load layers: export for test only
void load_conv(conv_struct *conv_layer, FILE* file_ptr, long int *file_idx);
void load_maxpool(maxpool_struct *maxpool_layer, FILE* file_ptr, long int *file_idx);
void load_flatten(flatten_struct *flatten_layer, FILE* file_ptr, long int *file_idx);
void load_dense(dense_struct *dense_layer, FILE* file_ptr, long int *file_idx);

void loadNet(net* loaded_net, FILE* file_ptr, long int* file_idx);

// executeNet.c

float* executeNet(net* net, generic_images imgs);

generic_images conv(generic_images input_images, conv_struct *parameters);
generic_images maxpool(generic_images input_images, maxpool_struct *parameters);
float* flatten(generic_images input_images, flatten_struct* flatten); 
float* dense(float* in_vect, dense_struct* dense); 
float relu(float x); 
void softmax(float* x, uint8_t dim_x); 

// prints.c

void print_conv_kernels(conv_struct* conv_layer);
void print_conv_bias(conv_struct* conv_layer);
void print_dense_kernel(dense_struct* dense_layer);
void print_dense_bias(dense_struct* dense_layer);
void print_images(generic_images res, int* dim);
void print_vect(float* res, uint8_t n);

// create_images.c
generic_images create_images_dif_val_row(int n_images, int image_size); 
generic_images create_images(int n_images, int image_size, int init_value);
float* arange(int len);

// frees.c
void free_data_set(dataset datas, int n_used, int n_generic_images, int n_generic_image, int n_generic_row);
void free_generic_images(generic_images imgs, int n_generic_image, int n_generic_row);
void free_dense_layer(dense_struct* dense_layer); 
void free_net(net* net_free);

void free_conv_layer(conv_struct* conv_layer);

// test.c

// test layer function
void conv_test();
void maxpool_test();
void flatten_test(); 
void dense_relu_test(); 
void dense_soft_test(); 

// test load layer function
void load_conv_test();
void load_dense_relu_test();
void load_flatten_test(); 
void load_maxpool_test(); 
void load_dense_softmax_test();

// test load with layer
void conv_completo_test();
void maxpool_test_completo(); 
void flatten_test_completo(); 
void dense_relu_test_completo(); 
void dense_softmax_test_completo(); 

// test all
void test_execution(); // test each layer execution
void test_load_layers_all(); // test every load
void test_completos(); // test each layer load and execution
void load_net_test_completo(); // test loadNet()
void execute_net_test_completo(); // test executeNet()

// loadNumpyImg.c

void load_image(generic_images images, FILE* image_ptr, long int* image_idx);
void load_vect(float* vect, FILE* image_ptr, long int* image_idx);

//measure_time.c 
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>

typedef unsigned long long ticks;
uint64_t readTSC(); 
void measure_rdtsc_all_layers(net* loaded_net, generic_images imgs);
void measure_rdtsc_by_layer_vect(net* loaded_net, generic_images imgs);
void measure_rdtsc_by_layer(net* loaded_net, generic_images imgs);
void measure_time_clock(net* loaded_net, generic_images imgs); 
void measure_gettimeofday_linux(net* loaded_net, generic_images imgs); 
void measure_gettimeprocess_linux(net* loaded_net, generic_images imgs);
void measure_gettimerealtime_linux(net* loaded_net, generic_images imgs); 

#endif