#pragma once
#include <arm_neon.h>
#include <cstdint>

#define Y_R 19595  // 0.299 * 65536
#define Y_G 38470  // 0.587 * 65536
#define Y_B 7471   // 0.114 * 65536

#define U_R (-11058) // -0.14713 * 65536
#define U_G (-21709) // -0.28886 * 65536
#define U_B 32767   // 0.436 * 65536

#define V_R 32767   // 0.615 * 65536
#define V_G (-27439) // -0.51499 * 65536
#define V_B (-5328)  // -0.10001 * 65536

#define CLAMP(x) (x < 0 ? 0 : (x > 255 ? 255 : x))

struct ConvertContext
{
    alignas(16) int16_t r_arr[8];
    alignas(16) int16_t g_arr[8];
    alignas(16) int16_t b_arr[8];
    alignas(16) uint8_t y_temp[8];
    alignas(16) uint8_t u_temp[8];
    alignas(16) uint8_t v_temp[8];
};

/**
 * Convert RGB24 to YUYV422 using NEON. Using the following formulas:
 *
 * 1. Y = 0.299R + 0.587G + 0.114B
 * 2. U = -0.14713R - 0.28886G + 0.436B
 * 3. V = 0.615R - 0.51499G - 0.10001B
 *
 * Fixed-point arithmetic optimization is used to speed up the calculation.
 *
 * Note:
 * 1. Process 8 pixels at a time (24 bytes input, 16 bytes output).
 * 2. Y component uses unsigned arithmetic, U/V components use signed arithmetic.
 *
 * @param rgb24 RGB24 data
 * @param yuyv422 YUYV422 data (output)
 * @param width Image width
 * @param height Image height
 * @param ctx Context for temporary data
 */
inline void convert_rgb24_to_yuyv_neon(const unsigned char* rgb24, unsigned char* yuyv422,
                                       const unsigned int width, const unsigned int height, ConvertContext& ctx)
{
    const unsigned int total_pixels = width * height;
    unsigned int index_yuyv = 0;
    unsigned int num_iter = total_pixels / 8;

    for (unsigned int i = 0; i < num_iter; i++)
    {
        uint8x8x3_t rgb = vld3_u8(rgb24);
        rgb24 += 24;

        uint16x8_t r16_u = vmovl_u8(rgb.val[0]);
        uint16x8_t g16_u = vmovl_u8(rgb.val[1]);
        uint16x8_t b16_u = vmovl_u8(rgb.val[2]);

        uint32x4_t y_low = vaddq_u32(
            vaddq_u32(vmull_n_u16(vget_low_u16(r16_u), Y_R),
                      vmull_n_u16(vget_low_u16(g16_u), Y_G)),
            vmull_n_u16(vget_low_u16(b16_u), Y_B));
        uint32x4_t y_high = vaddq_u32(
            vaddq_u32(vmull_n_u16(vget_high_u16(r16_u), Y_R),
                      vmull_n_u16(vget_high_u16(g16_u), Y_G)),
            vmull_n_u16(vget_high_u16(b16_u), Y_B));
        uint16x4_t y_low_u16 = vshrn_n_u32(y_low, 16);
        uint16x4_t y_high_u16 = vshrn_n_u32(y_high, 16);
        uint16x8_t y_u16 = vcombine_u16(y_low_u16, y_high_u16);
        uint8x8_t y_u8 = vmovn_u16(y_u16);

        int16x8_t r16 = vreinterpretq_s16_u16(r16_u);
        int16x8_t g16 = vreinterpretq_s16_u16(g16_u);
        int16x8_t b16 = vreinterpretq_s16_u16(b16_u);

        vst1q_s16(ctx.r_arr, r16);
        vst1q_s16(ctx.g_arr, g16);
        vst1q_s16(ctx.b_arr, b16);

        int16_t r_even_arr[4] = {ctx.r_arr[0], ctx.r_arr[2], ctx.r_arr[4], ctx.r_arr[6]};
        int16_t g_even_arr[4] = {ctx.g_arr[0], ctx.g_arr[2], ctx.g_arr[4], ctx.g_arr[6]};
        int16_t b_even_arr[4] = {ctx.b_arr[0], ctx.b_arr[2], ctx.b_arr[4], ctx.b_arr[6]};
        int16x4_t r_even = vld1_s16(r_even_arr);
        int16x4_t g_even = vld1_s16(g_even_arr);
        int16x4_t b_even = vld1_s16(b_even_arr);

        int32x4_t u_val = vaddq_s32(
            vaddq_s32(vmull_n_s16(r_even, U_R),
                      vmull_n_s16(g_even, U_G)),
            vmull_n_s16(b_even, U_B));
        u_val = vshrq_n_s32(u_val, 16);
        u_val = vmaxq_s32(u_val, vdupq_n_s32(0));
        u_val = vminq_s32(u_val, vdupq_n_s32(255));
        u_val = vaddq_s32(u_val, vdupq_n_s32(128));
        uint16x4_t u_u16 = vqmovun_s32(u_val);
        uint8x8_t u_u8 = vmovn_u16(vcombine_u16(u_u16, u_u16));

        int32x4_t v_val = vaddq_s32(
            vaddq_s32(vmull_n_s16(r_even, V_R),
                      vmull_n_s16(g_even, V_G)),
            vmull_n_s16(b_even, V_B));
        v_val = vshrq_n_s32(v_val, 16);
        v_val = vmaxq_s32(v_val, vdupq_n_s32(0));
        v_val = vminq_s32(v_val, vdupq_n_s32(255));
        v_val = vaddq_s32(v_val, vdupq_n_s32(128));
        uint16x4_t v_u16 = vqmovun_s32(v_val);
        uint8x8_t v_u8 = vmovn_u16(vcombine_u16(v_u16, v_u16));

        vst1_u8(ctx.y_temp, y_u8);
        vst1_u8(ctx.u_temp, u_u8);
        vst1_u8(ctx.v_temp, v_u8);

        for (int j = 0; j < 4; j++)
        {
            yuyv422[index_yuyv + 4 * j + 0] = ctx.y_temp[2 * j];
            yuyv422[index_yuyv + 4 * j + 1] = ctx.u_temp[j];
            yuyv422[index_yuyv + 4 * j + 2] = ctx.y_temp[2 * j + 1];
            yuyv422[index_yuyv + 4 * j + 3] = ctx.v_temp[j];
        }
        index_yuyv += 16;
    }

    unsigned int remaining = total_pixels % 8;
    for (int i = 0; i < remaining; i += 2)
    {
        const unsigned char r0 = rgb24[0];
        const unsigned char g0 = rgb24[1];
        const unsigned char b0 = rgb24[2];
        const unsigned char r1 = rgb24[3];
        const unsigned char g1 = rgb24[4];
        const unsigned char b1 = rgb24[5];

        const unsigned char y0 = CLAMP((Y_R * r0 + Y_G * g0 + Y_B * b0) >> 16);
        const unsigned char y1 = CLAMP((Y_R * r1 + Y_G * g1 + Y_B * b1) >> 16);
        const unsigned char u = CLAMP((U_R * r0 + U_G * g0 + U_B * b0) >> 16) + 128;
        const unsigned char v = CLAMP((V_R * r0 + V_G * g0 + V_B * b0) >> 16) + 128;

        yuyv422[index_yuyv + 0] = y0;
        yuyv422[index_yuyv + 1] = u;
        yuyv422[index_yuyv + 2] = y1;
        yuyv422[index_yuyv + 3] = v;
        rgb24 += 6;
        index_yuyv += 4;
    }
}
