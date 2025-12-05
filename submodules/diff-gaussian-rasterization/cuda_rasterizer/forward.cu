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

#include "forward.h"
#include "auxiliary.h"
#include "pcg32.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Macro for defining sample counts
#define SAMPLE_COUNTS \
    X(1) X(2) X(4) X(8) X(16) X(32) X(64) X(128) X(256) X(512) X(1024)

// Macro for template instantiations with float4 depths and uint4 corners
#define INSTANTIATE_RENDER_STOCHASTIC_POPFREE(SAMPLES) \
template __global__ void __launch_bounds__(BLOCK_X_L * BLOCK_Y_L * 1) \
renderStochastic_CUDA<NUM_CHANNELS, SAMPLES##u>( \
    const uint2* __restrict__, const uint32_t* __restrict__, \
    int, int, const float2* __restrict__, const float* __restrict__, \
    const float4* __restrict__, const float4* __restrict__, const uint4* __restrict__, \
    const float* __restrict__, const int, const float* __restrict__, \
    float* __restrict__);


#define INSTANTIATE_RENDER_STOCHASTIC(SAMPLES) \
template __global__ void __launch_bounds__(BLOCK_X_L * BLOCK_Y_L * 1) \
renderStochastic_CUDA<NUM_CHANNELS, SAMPLES##u>( \
    const uint2* __restrict__, const uint32_t* __restrict__, \
    int, int, const float2* __restrict__, const float* __restrict__, \
    const float4* __restrict__, const float* __restrict__, \
    const float* __restrict__, const int, const float* __restrict__, \
    float* __restrict__);

// Macro for switch case generation
#define RENDER_CASE(SAMPLES) \
    case SAMPLES: \
        renderStochastic_CUDA<NUM_CHANNELS, SAMPLES##u><<<grid, block>>>( \
            ranges, point_list, W, H, means2D, colors, conic_opacity, \
            depths, corners, cov3d, rand_seed, bg_color, out_color); \
        break;

#define RENDER_CASE_NO_CORNERS(SAMPLES) \
    case SAMPLES: \
        renderStochastic_CUDA<NUM_CHANNELS, SAMPLES##u><<<grid, block>>>( \
            ranges, point_list, W, H, means2D, colors, conic_opacity, \
            depths, cov3d, rand_seed, bg_color, out_color); \
        break;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// Helper structure to hold common preprocessing results
struct PreprocessCommon {
	float3 p_view;
	float3 p_orig;
	float3 p_proj;
	float3 conic;
	float opacity;
	float2 v1, v2;
	float lambda1, lambda2;
	float max_radius_x, max_radius_y;
	float2 point_image;
	uint2 rect_min, rect_max;
	const float* cov3D;
	bool valid;
};

// Common preprocessing logic shared by both variants
__device__ PreprocessCommon preprocessCommon(
	int idx, int P, int D, int M, int C,
	const float *__restrict__ orig_points,
	const glm::vec3 *__restrict__ scales,
	const float scale_modifier,
	const glm::vec4 *rotations,
	const float *opacities,
	const float *shs,
	bool *clamped,
	const float *cov3D_precomp,
	const float *colors_precomp,
	const float *viewmatrix,
	const float *fullprojmatrix,
	const glm::vec3 *cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float *cov3Ds,
	float *rgb,
	const dim3 grid,
	int block_x, int block_y,
	bool prefiltered)
{
	PreprocessCommon result;
	result.valid = false;

	// Perform near culling
	if (!in_frustum(idx, orig_points, viewmatrix, fullprojmatrix, prefiltered, result.p_view))
		return result;

	result.opacity = __ldg(&opacities[idx]);
	
	// Transform point by projecting
	result.p_orig = make_float3(__ldg(&orig_points[3 * idx]), 
								__ldg(&orig_points[3 * idx + 1]), 
								__ldg(&orig_points[3 * idx + 2]));
	float4 p_hom = transformPoint4x4(result.p_orig, fullprojmatrix);
	float p_w = 1.0f / fmaxf(p_hom.w, 0.0000001f);
	result.p_proj = make_float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);

	// Get or compute 3D covariance matrix
	if (cov3D_precomp != nullptr)
	{
		result.cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		result.cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(result.p_orig, focal_x, focal_y, tan_fovx, tan_fovy, result.cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return result;
	float det_inv = 1.f / det;
	result.conic = make_float3(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

	// Compute eigenvalues and eigenvectors
	float half_trace = (cov.x + cov.z) * 0.5f;
	float discriminant = half_trace * half_trace - det;
	discriminant = fmaxf(discriminant, 0.0f);
	float sqrt_discriminant = sqrtf(discriminant);
	result.lambda1 = half_trace + sqrt_discriminant;
	result.lambda2 = half_trace - sqrt_discriminant;
	if (result.lambda1 < 0.0f || result.lambda2 < 0.0f)
		return result;

	if (fabsf(cov.y) < 1e-6f) {
		result.v1 = make_float2(1.0f, 0.0f);
		result.v2 = make_float2(0.0f, 1.0f);
	} else {
		result.v1 = make_float2(result.lambda1 - cov.z, cov.y);
		float invNorm1 = rsqrtf(result.v1.x * result.v1.x + result.v1.y * result.v1.y);
		result.v1.x *= invNorm1;
		result.v1.y *= invNorm1;
		
		result.v2 = make_float2(-result.v1.y, result.v1.x);
	}
	
	const float k = min(2 * log(result.opacity * 255.0f), 3.3f);
	if (k <= 0)
		return result;
	float sigma1 = k * sqrtf(result.lambda1);
	float sigma2 = k * sqrtf(result.lambda2);

	float proj_x1 = sigma1 * fabsf(result.v1.x);
	float proj_x2 = sigma2 * fabsf(result.v2.x);
	float proj_y1 = sigma1 * fabsf(result.v1.y);
	float proj_y2 = sigma2 * fabsf(result.v2.y);
	result.max_radius_x = fmaxf(1.f, ceilf(fmaxf(proj_x1, proj_x2)));
	result.max_radius_y = fmaxf(1.f, ceilf(fmaxf(proj_y1, proj_y2)));
	result.point_image = make_float2(ndc2Pix(result.p_proj.x, W), ndc2Pix(result.p_proj.y, H));
	
	getRectAABB(result.point_image, result.max_radius_x, result.max_radius_y, 
				result.rect_min, result.rect_max, grid, block_x, block_y);
	if ((result.rect_max.x - result.rect_min.x) * (result.rect_max.y - result.rect_min.y) == 0)
		return result;

	// Compute colors if needed
	if (colors_precomp == nullptr)
	{
		glm::vec3 color_result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = color_result.x;
		rgb[idx * C + 1] = color_result.y;
		rgb[idx * C + 2] = color_result.z;
	}

	result.valid = true;
	return result;
}

template <int C>
__global__ void preprocessStochasticCUDA(
	int block_x, int block_y,
	int P, int D, int M,
	const float *__restrict__ orig_points,
	const glm::vec3 *__restrict__ scales,
	const float scale_modifier,
	const glm::vec4 *rotations,
	const float *opacities,
	const float *shs,
	bool *clamped,
	const float *cov3D_precomp,
	const float *colors_precomp,
	const float *viewmatrix,
	const float *fullprojmatrix,
    const float *projmatrix,
	const glm::vec3 *cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int *radii,
	float2 *points_xy_image,
	float4 *depths,
	uint4 *corners,
	float *cov3Ds,
	float *rgb,
	float4 *conic_opacity,
	const dim3 grid,
	uint32_t *tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	radii[P + idx] = 0;
	tiles_touched[idx] = 0;

	// Run common preprocessing
	PreprocessCommon common = preprocessCommon(
		idx, P, D, M, C, orig_points, scales, scale_modifier, rotations,
		opacities, shs, clamped, cov3D_precomp, colors_precomp,
		viewmatrix, fullprojmatrix, cam_pos, W, H, tan_fovx, tan_fovy,
		focal_x, focal_y, cov3Ds, rgb, grid, block_x, block_y, prefiltered);
	
	if (!common.valid)
		return;

	// Compute bounding box for all pixels
	uint2 rect_min2, rect_max2;
	getRectAll(common.point_image, common.max_radius_x, common.max_radius_y, rect_min2, rect_max2, W, H);
	corners[idx].x = rect_min2.x;
	corners[idx].y = rect_max2.x;
	corners[idx].z = rect_min2.y;
	corners[idx].w = rect_max2.y;

	// Compute plane approximation for depth computation
	glm::mat3 inv_vcov3d = getViewCov3DInverse(viewmatrix, common.cov3D);
	glm::vec3 vpos = glm::vec3(common.p_view.x, common.p_view.y, common.p_view.z);
	glm::vec4 plane = approximatePlane(vpos, inv_vcov3d);

	// Compute NDC coordinates of bounding box corners
	float2 ndcCorners[4];
    ndcCorners[0] = make_float2(pix2ndc(rect_min2.x, (float)W), pix2ndc(rect_min2.y, (float)H));
    ndcCorners[1] = make_float2(pix2ndc(rect_max2.x, (float)W), pix2ndc(rect_min2.y, (float)H));
    ndcCorners[2] = make_float2(pix2ndc(rect_min2.x, (float)W), pix2ndc(rect_max2.y, (float)H));
    ndcCorners[3] = make_float2(pix2ndc(rect_max2.x, (float)W), pix2ndc(rect_max2.y, (float)H));

	// Lambda to compute depth at each corner
    auto computeDepth = [&](const float2 ndcCorner) -> float {
        float4 ndcPos  = make_float4(ndcCorner.x, ndcCorner.y, 0.0f, 1.0f);
        float4 projpos = mat4_mul_vec4(projmatrix, ndcPos);
        
        float wInv = 1.0f / projpos.w;
        float3 pos = make_float3(projpos.x * wInv, projpos.y * wInv, projpos.z * wInv);
                                
        float invLen = rsqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
        float3 ray = make_float3(pos.x * invLen, pos.y * invLen, pos.z * invLen);
        
        float denom = plane.x * ray.x + plane.y * ray.y + plane.z * ray.z;
		if (fabsf(denom) < 1e-8f) return -1.0f;
        float t = -(plane.w) / denom;
        return (t < 0.0f) ? -1.0f : (t * ray.z);
    };

	// Compute all corner depths
    float depth0 = computeDepth(ndcCorners[0]);
	float depth1 = computeDepth(ndcCorners[1]);
	float depth2 = computeDepth(ndcCorners[2]);
	float depth3 = computeDepth(ndcCorners[3]);

	// Reject if any corner has invalid depth
	if (depth0 < 0.0f || depth1 < 0.0f || depth2 < 0.0f || depth3 < 0.0f)
		return;

	depths[idx] = make_float4(depth0, depth1, depth2, depth3);

	// Store results
	radii[idx] = common.max_radius_x;
	radii[idx + P] = common.max_radius_y;
	points_xy_image[idx] = common.point_image;
	conic_opacity[idx] = {common.conic.x, common.conic.y, common.conic.z, common.opacity};
	tiles_touched[idx] = (common.rect_max.y - common.rect_min.y) * (common.rect_max.x - common.rect_min.x);
}

template <int C>
__global__ void preprocessStochasticCUDA(
	int block_x, int block_y,
	int P, int D, int M,
	const float *__restrict__ orig_points,
	const glm::vec3 *__restrict__ scales,
	const float scale_modifier,
	const glm::vec4 *__restrict__ rotations,
	const float *__restrict__ opacities,
	const float *__restrict__ shs,
	bool *clamped,
	const float *__restrict__ cov3D_precomp,
	const float *__restrict__ colors_precomp,
	const float *__restrict__ viewmatrix,
	const float *fullprojmatrix,
    const float *projmatrix,
	const glm::vec3 *cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int *radii,
	float2 *points_xy_image,
	float *depths,
	float *cov3Ds,
	float *rgb,
	float4 *conic_opacity,
	const dim3 grid,
	uint32_t *tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0
	radii[idx] = 0;
	radii[idx + P] = 0;
	tiles_touched[idx] = 0;

	// Run common preprocessing
	PreprocessCommon common = preprocessCommon(
		idx, P, D, M, C, orig_points, scales, scale_modifier, rotations,
		opacities, shs, clamped, cov3D_precomp, colors_precomp,
		viewmatrix, fullprojmatrix, cam_pos, W, H, tan_fovx, tan_fovy,
		focal_x, focal_y, cov3Ds, rgb, grid, block_x, block_y, prefiltered);
	
	if (!common.valid)
		return;

	// Store results (simpler version without corner depth computation)
	depths[idx] = common.p_view.z;
	radii[idx] = common.max_radius_x;
	radii[idx + P] = common.max_radius_y;
	points_xy_image[idx] = common.point_image;
	conic_opacity[idx] = {common.conic.x, common.conic.y, common.conic.z, common.opacity};
	tiles_touched[idx] = (common.rect_max.y - common.rect_min.y) * (common.rect_max.x - common.rect_min.x);
}

template <uint8_t CHANNELS, int SAMPLES>
__global__ void __launch_bounds__(BLOCK_X_L *BLOCK_Y_L)
renderStochastic_CUDA( 
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ point_list,
        int W, int H,
        const float2 *__restrict__ points_xy_image,
        const float *__restrict__ features,
        const float4 *__restrict__ conic_opacity,
        const float *__restrict__ depths,
        const float *__restrict__ covs,
        int rand_seed,
        const float *__restrict__ bg_color,
        float *__restrict__ out_color)
{
    auto block = cg::this_thread_block();
    
    uint32_t horizontal_blocks = (W + BLOCK_X_L - 1) / BLOCK_X_L;
    uint2 pix_min = {block.group_index().x * BLOCK_X_L, block.group_index().y * BLOCK_Y_L};
    uint2 pix_max = {min(pix_min.x + BLOCK_X_L, W), min(pix_min.y + BLOCK_Y_L, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE_L - 1) / BLOCK_SIZE_L);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.    
    __shared__ int    collected_id[BLOCK_SIZE_L];
    __shared__ float2 collected_xy[BLOCK_SIZE_L];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE_L];
    __shared__ float  collected_depth[BLOCK_SIZE_L];
    // __shared__ float collected_colors[CHANNELS*BLOCK_SIZE_L];

    pcg32_state rng;
    pcg32_srandom(&rng, sample_tea_32(pix_id, rand_seed), rand_seed);
    // curandState state;
    // curand_init(sample_tea_32(pix_id, rand_seed), 0, 0, &state);

    float this_z[SAMPLES];
    float this_c[SAMPLES * CHANNELS];
    for (int si = 0; si < SAMPLES; ++si){
        this_z[si] = FLT_MAX;
        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ++ch)
            this_c[si * CHANNELS + ch] = (bg_color[ch]);
    }


    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE_L){
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE_L)
            return;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE_L + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()]            = coll_id;
            collected_xy[block.thread_rank()]            = __ldg(&points_xy_image[coll_id]);
            collected_conic_opacity[block.thread_rank()] = __ldg(&conic_opacity[coll_id]);
            collected_depth[block.thread_rank()]         = __ldg(&depths[coll_id]);

            // for (int ch = 0; ch < CHANNELS; ++ch)
            //  collected_colors[block.thread_rank() * CHANNELS + ch] = (features[coll_id * CHANNELS + ch]);
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE_L, toDo); j++)
        {
            float zval = __ldg(&collected_depth[j]);

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j];
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

            if (power > 0.0f)
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix).
            float alpha = con_o.w * exp(power);
            if (alpha < 1.0f / 255.0f)
                continue;
                
			for (int si = 0; si < SAMPLES; ++si)
				{
					float uniform_val = pcg32_float(&rng);
					if ((uniform_val < alpha) && (collected_depth[j] < this_z[si]))
					{
						this_z[si] = collected_depth[j];
						#pragma unroll
						for (int ch = 0; ch < CHANNELS; ++ch) {
							this_c[si * CHANNELS + ch] =
								(features[collected_id[j] * CHANNELS + ch]);
						}
					}
				}
        }
    }

    if (!inside)
        return;

    const int img_size = W * H;
    int base = pix_id;
    constexpr float inv_samples = 1.0f / SAMPLES;

    #pragma unroll
    for (int ch = 0; ch < CHANNELS; ++ch) {
        float temp_c = 0.0f;
        for (int si = 0; si < SAMPLES; ++si) {
            temp_c += this_c[si * CHANNELS + ch];
        }
        out_color[base] = temp_c * inv_samples;
        base += img_size;
    }
}



template <uint8_t CHANNELS, uint16_t SAMPLES>
__global__ void __launch_bounds__(BLOCK_X_L *BLOCK_Y_L * 1)
	renderStochastic_CUDA( // 
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ features,
		const float4 *__restrict__ conic_opacity,
		const float4 *__restrict__ depths,
		const uint4 *__restrict__ corners,
		const float *__restrict__ covs,
		int rand_seed,
		const float *__restrict__ bg_color,
		float *__restrict__ out_color)
{
	auto block = cg::this_thread_block();

	// unsigned int sind = block.thread_index().x % 1;
	unsigned int xind = block.thread_index().x;
	
	uint32_t horizontal_blocks = (W + BLOCK_X_L - 1) / BLOCK_X_L;
	uint2 pix_min = {block.group_index().x * BLOCK_X_L, block.group_index().y * BLOCK_Y_L};
	uint2 pix_max = {min(pix_min.x + BLOCK_X_L, W), min(pix_min.y + BLOCK_Y_L, H)};
	uint2 pix = {pix_min.x + xind, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE_L- 1) / BLOCK_SIZE_L);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.	
	__shared__ int collected_id[BLOCK_SIZE_L];
	__shared__ float2 collected_xy[BLOCK_SIZE_L];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE_L];
	__shared__ float4 collected_depth[BLOCK_SIZE_L];
	__shared__ uint4 collected_corners[BLOCK_SIZE_L];
	__shared__ half collected_colors[CHANNELS*BLOCK_SIZE_L];

	pcg32_state rng;
	pcg32_srandom(&rng, sample_tea_32(pix_id, rand_seed), rand_seed);

	float this_z[SAMPLES];
	half this_c[SAMPLES * CHANNELS];
	for (uint16_t si = 0; si < SAMPLES; ++si){
		this_z[si] = (FLT_MAX);
		#pragma unroll
		for (uint8_t ch = 0; ch < CHANNELS; ++ch)
			this_c[si * CHANNELS + ch] = __float2half(bg_color[ch]);
	}

	// End if entire block votes that it is done rasterizing
	uint16_t num_done = __syncthreads_count(done);
	if (num_done == BLOCK_SIZE_L)
		return;

	for (uint16_t i = 0; i < rounds; i++, toDo -= BLOCK_SIZE_L)
	{
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE_L + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = __ldg(&points_xy_image[coll_id]);
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = __ldg(&depths[coll_id]);
			collected_corners[block.thread_rank()] = __ldg(&corners[coll_id]);

			for (uint8_t ch = 0; ch < CHANNELS; ++ch)
				collected_colors[block.thread_rank() * CHANNELS + ch] = __float2half(features[coll_id * CHANNELS + ch]);
		}
		block.sync();

		// Iterate over current batch
		for (uint16_t j = 0; !done && j < min(BLOCK_SIZE_L, toDo); j++)
		{
			bool inside_corners = (pix.x >= collected_corners[j].x) &&
                      (pix.x <= collected_corners[j].y) &&
                      (pix.y >= collected_corners[j].z) &&
                      (pix.y <= collected_corners[j].w);
			if (!inside_corners) continue;

			float zval = bilinearInterpolateKernel(pix.x, pix.y, collected_corners[j], collected_depth[j]);

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
				
			for (uint16_t si = 0; si < SAMPLES; ++si)
			{
				float uniform_val = pcg32_float(&rng);
				if ((uniform_val < alpha) && (zval < this_z[si]))
				{
					this_z[si] = zval;
					#pragma unroll
					for (uint8_t ch = 0; ch < CHANNELS; ++ch){
						this_c[si * CHANNELS + ch] = collected_colors[j * CHANNELS + ch];
					}
				}
			}
		}
	}

	if (inside)
	{
		const int img_size = W * H;
		int base = pix_id;
		constexpr float inv_samples = 1.0f / SAMPLES;
		
		#pragma unroll
		for (uint8_t ch = 0; ch < CHANNELS; ++ch) {
			float temp_c = 0.0f;
			for (uint16_t si = 0; si < SAMPLES; ++si)
				temp_c += __half2float(this_c[si * CHANNELS + ch]);
			out_color[base] = temp_c * inv_samples;
			base += img_size;
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

void FORWARD::preprocess_stochastic_popfree(
	int block_x, int block_y,
	int P, int D, int M,
	const float *means3D,
	const glm::vec3 *scales,
	const float scale_modifier,
	const glm::vec4 *rotations,
	const float *opacities,
	const float *shs,
	bool *clamped,
	const float *cov3D_precomp,
	const float *colors_precomp,
	const float *viewmatrix,
	const float *fullprojmatrix,
	const float *projmatrix,
	const glm::vec3 *cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int *radii,
	float2 *means2D,
	float4 *depths,
	uint4 *corners,
	float *cov3Ds,
	float *rgb,
	float4 *conic_opacity,
	const dim3 grid,
	uint32_t *tiles_touched,
	bool prefiltered)
{
	preprocessStochasticCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		block_x, block_y,
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		fullprojmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		corners,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}

void FORWARD::preprocess_stochastic(
	int block_x, int block_y,
	int P, int D, int M,
	const float *means3D,
	const glm::vec3 *scales,
	const float scale_modifier,
	const glm::vec4 *rotations,
	const float *opacities,
	const float *shs,
	bool *clamped,
	const float *cov3D_precomp,
	const float *colors_precomp,
	const float *viewmatrix,
	const float *fullprojmatrix,
	const float *projmatrix,
	const glm::vec3 *cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int *radii,
	float2 *means2D,
	float *depths,
	float *cov3Ds,
	float *rgb,
	float4 *conic_opacity,
	const dim3 grid,
	uint32_t *tiles_touched,
	bool prefiltered)
{
	preprocessStochasticCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		block_x, block_y,
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		fullprojmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}

void FORWARD::render_stochastic_popfree(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int num_samples,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	const float4* depths,
	const uint4* corners,
	const float* cov3d,
	const int rand_seed,
	const float* bg_color,
	float* out_color)
{
	switch (num_samples)
	{
#define X(SAMPLES) RENDER_CASE(SAMPLES)
		SAMPLE_COUNTS
#undef X
		default:
			break;
	}
}

#define X(SAMPLES) INSTANTIATE_RENDER_STOCHASTIC_POPFREE(SAMPLES)
SAMPLE_COUNTS
#undef X



void FORWARD::render_stochastic(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int num_samples,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	const float* depths,
	const float* cov3d,
	const int rand_seed,
	const float* bg_color,
	float* out_color)
{
	switch (num_samples)
	{
#define X(SAMPLES) RENDER_CASE_NO_CORNERS(SAMPLES)
		SAMPLE_COUNTS
#undef X
		default:
			break;
	}
}

#define X(SAMPLES) INSTANTIATE_RENDER_STOCHASTIC(SAMPLES)
SAMPLE_COUNTS
#undef X
