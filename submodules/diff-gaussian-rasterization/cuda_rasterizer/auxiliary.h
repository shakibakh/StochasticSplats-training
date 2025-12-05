/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <cuda_fp16.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define BLOCK_SIZE_L (BLOCK_X_L * BLOCK_Y_L)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float pix2ndc(float pix, int S) 
{
    return ((pix * 2.0f + 1.0f) / S) - 1.0f;
}

__forceinline__ __device__ uint32_t sl(uint32_t x, int n)
{
	return x << n;
}

__forceinline__ __device__ uint32_t sr(uint32_t x, int n)
{
	return x >> n;
}

__forceinline__ __device__ uint32_t sample_tea_32(uint32_t v0, uint32_t v1, int rounds = 32)
{
	uint32_t sum = 0;
	for (int i = 0; i < rounds; ++i)
	{
		sum += 0x9e3779b9;
		v0 += (sl(v1, 4) + 0xa341316c) ^ (v1 + sum) ^ (sr(v1, 5) + 0xc8013ea4);
		v1 += (sl(v0, 4) + 0xad90777d) ^ (v0 + sum) ^ (sr(v0, 5) + 0x7e95761e);
	}
	return v0;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRectAABB(const float2 p, int max_radius_x, int max_radius_y, uint2& rect_min, uint2& rect_max, dim3 grid, int block_x, int block_y)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius_x) / block_x))),
		min(grid.y, max((int)0, (int)((p.y - max_radius_y) / block_y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius_x + block_x - 1) / block_x))),
		min(grid.y, max((int)0, (int)((p.y + max_radius_y + block_y - 1) / block_y)))
	};
}

__forceinline__ __device__ void getRectAll(const float2 p, int max_radius_x, int max_radius_y, uint2& rect_min, uint2& rect_max, int W, int H)
{
	rect_min = {
		min((uint)W, max((int)0, (int)((p.x - max_radius_x)))),
		min((uint)H, max((int)0, (int)((p.y - max_radius_y))))
	};
	rect_max = {
		min((uint)W, max((int)0, (int)((p.x + max_radius_x)))),
		min((uint)H, max((int)0, (int)((p.y + max_radius_y))))
	};
}

__forceinline__ __device__ float bilinearInterpolateKernel(
    const uint x_star,
    const uint y_star,
    const uint4 corners,
    const float4 zs
)
{
	float denomX = 1.0f / (corners.y - corners.x);
    float denomY = 1.0f / (corners.w - corners.z);

    float u = (x_star - corners.x) * denomX;
    float v = (y_star - corners.z) * denomY;

	float z_val =
		(1.0f - u) * (1.0f - v) * zs.x +
			u      * (1.0f - v) * zs.y +
			u      *       v    * zs.w +
		(1.0f - u) *       v    * zs.z;

	return z_val;
}

__forceinline__ __device__ glm::vec4 approximatePlane(glm::vec3 mean, glm::mat3 inv_vcov3d){
    glm::vec3 gradient = inv_vcov3d * mean;
    float d = -dot(gradient, mean);
    return glm::vec4(gradient, d);
}

__forceinline__ __device__ glm::mat3 getViewCov3DInverse(const float *viewmatrix, const float *cov3D)
{
    // Extract the 3×3 portion "W" from the view matrix:
    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]
    );

    // Rebuild the 3×3 covariance from the 6 unique elements:
    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    );
    glm::mat3 invVrk = glm::inverse(Vrk);
    glm::mat3 invCov = W * invVrk * glm::transpose(W);

    return invCov;
}


__device__ __forceinline__ float4 mat4_mul_vec4(const float* M, const float4 v)
{
    // M is assumed row-major, 16 elements.
    // If your matrix is column-major, transpose indexing accordingly.
    return make_float4(
        M[0] * v.x + M[4] * v.y + M[8]  * v.z + M[12] * v.w,
        M[1] * v.x + M[5] * v.y + M[9]  * v.z + M[13] * v.w,
        M[2] * v.x + M[6] * v.y + M[10] * v.z + M[14] * v.w,
        M[3] * v.x + M[7] * v.y + M[11] * v.z + M[15] * v.w
    );
}

__device__ __forceinline__ float fixDenominator(float denom)
{
    float eps = 1e-6f;
    if (fabsf(denom) < eps) {
        eps = (denom < 0.0f) ? -eps : eps;
    }
    return denom + eps;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif