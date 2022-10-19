#ifndef __DENSE__
#define __DENSE__

#include "main.h"

float* dense(float* in_vect, dense_struct* dense){

    // Given a dense struct and an input vector
    // Return the result of the dense layer for the input vector 
    // with dense struct parameters

    float* res = malloc((dense->dim_out)*sizeof(float));  // output vector 
    
    // Calculate W*x
    // being W the dense->weights and x the input vector
    // where W n*m, x is m*1 and output n*1
    // with m = length(x) = rows(W) and n = columns(W)

    // for each output
    for (int i = 0; i < (dense->dim_out); i++) //n
    {
        res[i] = 0; 
        // for each input
        int rest_ind = (dense->dim_in)%4; 
        for (int j = 0;j < (dense->dim_in - rest_ind) ; j+=4) //m 
        {
            //load data into Vd registers 
            float32x4_t w_row, x_col;
			w_row = vld1q_f32( &(dense->weights[i][j]));
			x_col = vld1q_f32(&in_vect[j]);

            //multiply
			float32x4_t res_vect = vmulq_f32(w_row, x_col);

			//horizontal add
			float32x2_t high = vget_high_f32(res_vect);
			float32x2_t low = vget_low_f32(res_vect); 
			high = vadd_f32(high, low); 
			//add consecutives high
			high = vpadd_f32(high, high);
			
            //get sum
            res[i] += vget_lane_f32(high, 0);
             
              
        }  
        for (int j = 0; j < rest_ind; j++)
        {
            res[i] += (dense->weights[i][(dense->dim_in - rest_ind )+ j])* in_vect[j];
            
        }
    }

    // Add bias
    
    int rest_ind = (dense->dim_out)%4;
    for(int i = 0;i < (dense->dim_out - rest_ind); i+=4){
        
        //load vect and bias into Vd register 
        float32x4_t vect = vld1q_f32(&res[i]); 
        float32x4_t bias = vld1q_f32(&dense->bias[i]);
        
        float h[4];
        vst1q_f32(&h[0], bias);
        float32x4_t sum = vaddq_f32(vect, bias); //add
        
        //store into res
        vst1q_f32(&res[i], sum);  
    } 
    
    for (int i = 0; i < rest_ind ; i++)
    { 
        res[(dense->dim_out - rest_ind) + i] += dense->bias[(dense->dim_out - rest_ind ) + i];
         
    }
     
    // Apply activation function
    if((dense->actFunc[0] == 'r') && (dense->actFunc[1] == 'e') 
    && (dense->actFunc[2] == 'l') && (dense->actFunc[3] == 'u')){
        
        //relu case
        int rest = (dense->dim_out)%4; 
        for (int i = 0; i < (dense->dim_out - rest); i+=4)
        {
            float32x4_t v1 = vld1q_f32(&res[i]);
            float32x4_t v2 = vdupq_n_f32(0);  
            float32x4_t max = vmaxq_f32(v1, v2);  
            vst1q_f32(&res[i], max); 
        }
        
        for (int i = 0; i < (rest); i++)
        {
            res[(dense->dim_out -rest ) + i] = relu(res[(dense->dim_out -rest ) + i]); 
        }

    }else{

        //softmax case
        softmax(res, (dense->dim_out)); 
    }

    free(in_vect);
    
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

void softmax(float* x, uint8_t dim_x){

    // Given a float vector
    // Returns its softmax result

    float suma = 0; 
    for (int i = 0; i < dim_x; i++)
    {
        suma += exp(x[i]); 
    }
    for (int i = 0; i < dim_x; i++)
    {
        x[i] = exp(x[i]) / suma ;  
    }
     
}

#endif
