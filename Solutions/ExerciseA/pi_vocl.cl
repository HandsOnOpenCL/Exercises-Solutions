//------------------------------------------------------------------------------
//
// kernel:  pi    
//
// Purpose: accumulate partial sums of pi comp
// 
// input: float step_size
//        int   niters per work item
//        local float* an array to hold sums from each work item
//
// output: partial_sums   float vector of partial sums
//

__kernel void pi(                                       
   const int          niters,                           
   const float        step_size,                        
   __local  float*    local_sums,                       
   __global float*    partial_sums)                     
{                                                       
   int num_wrk_items  = get_local_size(0);              
   int local_id       = get_local_id(0);                
   int group_id       = get_group_id(0);                
   float x, sum, accum = 0.0f;                           
   int i,istart,iend;                                       
   istart = (group_id * num_wrk_items + local_id) * niters; 
   iend   = istart+niters;                 
   for(i= istart; i<iend; i++){            
       x = (i+0.5f)*step_size;              
       accum += 4.0f/(1.0f+x*x);             
   }                                       
   local_sums[local_id] = accum;           
   barrier(CLK_LOCAL_MEM_FENCE);           
   if (local_id == 0){                     
      sum = 0.0f;                           
      for(i=0; i<num_wrk_items;i++){       
          sum += local_sums[i];            
      }                                    
      partial_sums[group_id] = sum;        
   }                                       
}                                          

__kernel void pi_vec4(                     
   const int          niters,              
   const float        step_size,           
   __local  float*    local_sums,          
   __global float*    partial_sums)        
{                                          
   int num_wrk_items  = get_local_size(0); 
   int local_id       = get_local_id(0); 
   int group_id       = get_group_id(0); 
   float sum, accum = 0.0f;        
                                      
   float4 x, psum_vec;                
   float4 ramp={0.5f, 1.5f, 2.5f, 3.5f};  
   float4 four={4.0f, 4.0f, 4.0f, 4.0f};  
   float4 one ={1.0f, 1.0f, 1.0f, 1.0f};  
                                      
   int i,istart,iend;                 
   istart = (group_id * num_wrk_items + local_id) * niters; 
   iend   = istart+niters;                        
   for(i= istart; i<iend; i=i+4){                 
     x = ((float4)i+ramp)*step_size;              
     psum_vec=four/(one + x*x);                   
     accum += psum_vec.s0 + psum_vec.s1 + psum_vec.s2 + psum_vec.s3; 
   }                                     
   local_sums[local_id] = accum;         
   barrier(CLK_LOCAL_MEM_FENCE);          
   if (local_id == 0){                    
      sum = 0.0f;                          
      for(i=0; i<num_wrk_items;i++){      
          sum += local_sums[i];           
      }                                   
      partial_sums[group_id] = sum;       
   }                                      
}                                         

__kernel void pi_vec8(                    
   const int          niters,             
   const float        step_size,          
   __local  float*    local_sums,         
   __global float*    partial_sums)       
{                                         
   int num_wrk_items  = get_local_size(0);
   int local_id       = get_local_id(0);
   int group_id       = get_group_id(0);
   float sum, accum = 0.0f; 
                             
   float8 x, psum_vec;       
   float8 ramp={0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};     
   float8 four={4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};     
   float8 one ={1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};     
                                  
   int i,istart,iend;            
   istart = (group_id * num_wrk_items + local_id) * niters;
   iend   = istart+niters;           
   for(i= istart; i<iend; i=i+8){    
     x = ((float8)i+ramp)*step_size; 
     psum_vec=four/(one + x*x);      
     accum += psum_vec.s0 + psum_vec.s1 + psum_vec.s2 + psum_vec.s3 +
              psum_vec.s4 + psum_vec.s5 + psum_vec.s6 + psum_vec.s7; 
   }                                
   local_sums[local_id] = accum;    
   barrier(CLK_LOCAL_MEM_FENCE);    
   if (local_id == 0){              
      sum = 0.0f;                    
      for(i=0; i<num_wrk_items;i++){
          sum += local_sums[i];     
      }                             
      partial_sums[group_id] = sum; 
   }
}                     