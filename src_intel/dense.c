#ifndef __DENSE__
#define __DENSE__

#include "main.h"

float* dense(float* in_vect, dense_struct* dense){

    // Given a dense struct and an input vector
    // Return the result of the dense layer for the input vector 
    // with dense struct parameters
    
    
    float* res = _mm_malloc((dense->dim_out)*sizeof(float), 32);  // output vector 
    
    // Calculate W*x
    // being W the dense->weights and x the input vector
    // where W n*m, x is m*1 and output n*1

    // for each output
    for (int i = 0; i < (dense->dim_out); i++) //n
    {
        res[i] = 0; 
        
        // for each input
        int rest_ind = (dense->dim_in)%8; 
        for (int j = 0;j < (dense->dim_in - rest_ind) ; j+=8) //m 
        {
            
            //load data into ymm registers 
            __m256 w_row = _mm256_load_ps( &(dense->weights[i][j])); 
            __m256 x_col = _mm256_load_ps(&in_vect[j]); 
            
            __m256 mul = _mm256_mul_ps(w_row, x_col); 
            
            //horizontal add 
            __m128 vlow  = _mm256_castps256_ps128(mul); //cast lower pos as _m128 
            __m128 vhigh = _mm256_extractf128_ps(mul, 1); // high 128

            vlow  = _mm_add_ps(vlow, vhigh);     // add low and high -> 4 partial sums 

            __m128 shuf = _mm_movehdup_ps(vlow); //duplicate odd pos 

            __m128 sums = _mm_add_ps(vlow, shuf); //partials sums in even pos 

            shuf = _mm_movehl_ps(shuf, sums); // move partial sum to lower pos 

            sums = _mm_add_ss(sums, shuf); //add lower pos 

            res[i] += _mm_cvtss_f32(sums); //cast lower pos as float
            
            
        } 
        for (int j = 0; j < rest_ind; j++)
        {
            res[i] += (dense->weights[i][(dense->dim_in - rest_ind )+ j])* in_vect[j];
            
        }
    }
    
    // Add bias
    int rest_ind = (dense->dim_out)%8; 
    for(int i = 0;i < (dense->dim_out - rest_ind); i+=8){
        
        //load vect and bias into ymm register 
        __m256 vect = _mm256_load_ps(&res[i]); 
        __m256 bias = _mm256_load_ps(&dense->bias[i]);
        
        vect = _mm256_add_ps(vect, bias); //add
        //store into res
        _mm256_store_ps(&res[i], vect);  
    } 
    for (int i = 0; i < rest_ind ; i++)
    { 
        res[(dense->dim_out - rest_ind) + i] += dense->bias[(dense->dim_out - rest_ind) + i];
         
    }
    
    // Apply activation function
    if((dense->actFunc[0] == 'r') && (dense->actFunc[1] == 'e') 
    && (dense->actFunc[2] == 'l') && (dense->actFunc[3] == 'u')){
        
        //relu case
        int rest = (dense->dim_out)%8; 
        for (int i = 0; i < (dense->dim_out - rest); i+=8)
        {
            __m256 v1 = _mm256_load_ps(&res[i]);
            __m256 v2 = _mm256_setzero_ps();  
            __m256 max = _mm256_max_ps(v1, v2);  
            _mm256_store_ps(&res[i], max); 
        }
        
        for (int i = 0; i < (rest); i++)
        {
            res[(dense->dim_out -rest ) + i] = relu(res[(dense->dim_out -rest ) + i]); 
        }

    }else{

        //softmax case
        softmax(res, (dense->dim_out)); 
    }
    
    _mm_free(in_vect); 
    
    return res;
}

float relu(float x){

    // Given a float value
    // Return 0 if it is less than 0, otherwise returns the value
    if(x < 0){
        return 0;  
    }else{
        return x; 
    }
}

__m128 exp_vect(__m128 x){
    __m128 f, p, r;
    __m128i t, j;
    const __m128 a = _mm_set1_ps(12102203.0f); /* (1 << 23) / log(2) */
    const __m128i m = _mm_set1_epi32(0xff800000); /* mask for integer bits */
    const __m128 ttm23 = _mm_set1_ps(1.1920929e-7f); /* exp2(-23) */
    const __m128 c0 = _mm_set1_ps(0.3371894346f);
    const __m128 c1 = _mm_set1_ps(0.657636276f);
    const __m128 c2 = _mm_set1_ps(1.00172476f);

    t = _mm_cvtps_epi32(_mm_mul_ps (a, x));
    j = _mm_and_si128(t, m);            // j = (int)(floor (x/log(2))) << 23 
    t = _mm_sub_epi32(t, j);
    f = _mm_mul_ps(ttm23, _mm_cvtepi32_ps (t)); // f = (x/log(2)) - floor (x/log(2)) 
    p = c0;                              // c0 
    p = _mm_mul_ps(p, f);               // c0 * f 
    p = _mm_add_ps(p, c1);              // c0 * f + c1 
    p = _mm_mul_ps(p, f);               // (c0 * f + c1) * f 
    p = _mm_add_ps(p, c2);              // p = (c0 * f + c1) * f + c2 ~= 2^f 
    r = _mm_castsi128_ps(_mm_add_epi32(j, _mm_castps_si128(p))); // r = p * 2^i
    
    return r; 
}

void softmax(float* x, uint8_t dim_x){

    // Given a float vector
    // Returns its softmax result 

    float suma = 0; 
    int rest = (dim_x%4); 
    for (int i = 0; i < dim_x-rest; i+=4)
    {
        
        __m128 v = _mm_load_ps(&x[i]); 
        __m128 v_exp = exp_vect(v);
        //hor add
        __m128 sum = _mm_movehdup_ps(v_exp); //duolicate odd pos
        sum = _mm_add_ps(sum, v_exp); 
        v_exp = _mm_movehl_ps(sum, sum); //high ->low 
        
        suma += _mm_cvtss_f32(_mm_add_ss(sum, v_exp)); 
    } 
    for (int i = 0; i < rest; i++)
    {
        suma += (float)exp(x[dim_x -rest + i]); 
    }
     
    for (int i = 0; i < dim_x-rest; i+=4)
    {
        
        __m128 v = _mm_load_ps(&x[i]);
        __m128 v_exp = exp_vect(v); 
        __m128 sum = _mm_set_ps(suma, suma, suma, suma);
        v_exp = _mm_div_ps(v_exp, sum); 
        _mm_store_ps(&x[i], v_exp);    
    }
    for (int i = 0; i < rest; i++)
    {
        x[dim_x -rest + i] = (float)exp(x[ dim_x -rest + i]) / suma ;  
    } 
}

#endif