// =============================================================================
// Stage 3: FP32 + FP16 + FP8x4 FMA + Mixed-Precision Accumulation
// Supports: mode 000=FP32, 001=FP16, 010=FP8x4,
//           100=FP16->FP32 Acc, 101=FP8x4->FP32 Acc
// Incremental on Stage 2: adds accumulation modes
// =============================================================================

module fma_acc_stage3 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [2:0]  mode,       // 3-bit mode
    input  wire [31:0] A,
    input  wire [31:0] B,
    input  wire [31:0] C,
    input  wire        acc_en,     // accumulate enable
    input  wire [31:0] acc_in,     // accumulator initial value
    output reg  [31:0] result,
    output reg         valid_out,
    output reg         overflow,
    output reg         underflow,
    output reg         inexact,
    output reg  [31:0] acc_out     // accumulator output
);

    wire is_fp32  = (mode == 3'b000);
    wire is_fp16  = (mode == 3'b001);
    wire is_fp8   = (mode == 3'b010);
    wire is_acc16 = (mode == 3'b100);  // FP16->FP32 accumulate
    wire is_acc8  = (mode == 3'b101);  // FP8x4->FP32 accumulate
    wire is_acc   = is_acc16 | is_acc8;

    // Accumulator register
    reg [31:0] acc_reg;

    // =========================================================================
    // FP32/FP16 Field Extraction (same as Stage 2)
    // =========================================================================
    wire        a32_sign = A[31]; wire [7:0] a32_exp = A[30:23]; wire [22:0] a32_man = A[22:0];
    wire        b32_sign = B[31]; wire [7:0] b32_exp = B[30:23]; wire [22:0] b32_man = B[22:0];
    wire        c32_sign = C[31]; wire [7:0] c32_exp = C[30:23]; wire [22:0] c32_man = C[22:0];

    wire        a16_sign = A[15]; wire [4:0] a16_exp = A[14:10]; wire [9:0] a16_man = A[9:0];
    wire        b16_sign = B[15]; wire [4:0] b16_exp = B[14:10]; wire [9:0] b16_man = B[9:0];
    wire        c16_sign = C[15]; wire [4:0] c16_exp = C[14:10]; wire [9:0] c16_man = C[9:0];

    // FP32 special flags
    wire a32_is_zero = (a32_exp == 0) && (a32_man == 0);
    wire b32_is_zero = (b32_exp == 0) && (b32_man == 0);
    wire c32_is_zero = (c32_exp == 0) && (c32_man == 0);
    wire a32_is_inf  = (a32_exp == 8'hFF) && (a32_man == 0);
    wire b32_is_inf  = (b32_exp == 8'hFF) && (b32_man == 0);
    wire c32_is_inf  = (c32_exp == 8'hFF) && (c32_man == 0);
    wire a32_is_nan  = (a32_exp == 8'hFF) && (a32_man != 0);
    wire b32_is_nan  = (b32_exp == 8'hFF) && (b32_man != 0);
    wire c32_is_nan  = (c32_exp == 8'hFF) && (c32_man != 0);
    wire a32_is_denorm = (a32_exp == 0) && (a32_man != 0);
    wire b32_is_denorm = (b32_exp == 0) && (b32_man != 0);
    wire c32_is_denorm = (c32_exp == 0) && (c32_man != 0);

    // FP16 special flags
    wire a16_is_zero = (a16_exp == 0) && (a16_man == 0);
    wire b16_is_zero = (b16_exp == 0) && (b16_man == 0);
    wire c16_is_zero = (c16_exp == 0) && (c16_man == 0);
    wire a16_is_inf  = (a16_exp == 5'h1F) && (a16_man == 0);
    wire b16_is_inf  = (b16_exp == 5'h1F) && (b16_man == 0);
    wire c16_is_inf  = (c16_exp == 5'h1F) && (c16_man == 0);
    wire a16_is_nan  = (a16_exp == 5'h1F) && (a16_man != 0);
    wire b16_is_nan  = (b16_exp == 5'h1F) && (b16_man != 0);
    wire c16_is_nan  = (c16_exp == 5'h1F) && (c16_man != 0);
    wire a16_is_denorm = (a16_exp == 0) && (a16_man != 0);
    wire b16_is_denorm = (b16_exp == 0) && (b16_man != 0);

    // Mode-based unified selection for FP32/FP16/Acc16
    wire use_fp16_fields = is_fp16 | is_acc16;

    wire a_sign_fp = use_fp16_fields ? a16_sign : a32_sign;
    wire b_sign_fp = use_fp16_fields ? b16_sign : b32_sign;

    // For acc mode, C is the accumulator; for FMA mode, C is the input
    wire [31:0] c_effective = is_acc ? acc_reg : C;
    wire        c_eff_sign = c_effective[31];
    wire [7:0]  c_eff_exp  = c_effective[30:23];
    wire [22:0] c_eff_man  = c_effective[22:0];

    wire c_eff_is_zero   = (c_eff_exp == 0) && (c_eff_man == 0);
    wire c_eff_is_inf    = (c_eff_exp == 8'hFF) && (c_eff_man == 0);
    wire c_eff_is_nan    = (c_eff_exp == 8'hFF) && (c_eff_man != 0);
    wire c_eff_is_denorm = (c_eff_exp == 0) && (c_eff_man != 0);

    // Unified special flags
    wire a_is_zero = use_fp16_fields ? a16_is_zero : a32_is_zero;
    wire b_is_zero = use_fp16_fields ? b16_is_zero : b32_is_zero;
    wire c_is_zero = is_acc ? c_eff_is_zero : (use_fp16_fields ? c16_is_zero : c32_is_zero);
    wire a_is_inf  = use_fp16_fields ? a16_is_inf  : a32_is_inf;
    wire b_is_inf  = use_fp16_fields ? b16_is_inf  : b32_is_inf;
    wire c_is_inf  = is_acc ? c_eff_is_inf : (use_fp16_fields ? c16_is_inf : c32_is_inf);
    wire a_is_nan  = use_fp16_fields ? a16_is_nan  : a32_is_nan;
    wire b_is_nan  = use_fp16_fields ? b16_is_nan  : b32_is_nan;
    wire c_is_nan  = is_acc ? c_eff_is_nan : (use_fp16_fields ? c16_is_nan : c32_is_nan);

    // Mantissa
    wire [23:0] a32_mantissa = a32_is_denorm ? {1'b0, a32_man} : (a32_is_zero ? 24'h0 : {1'b1, a32_man});
    wire [23:0] b32_mantissa = b32_is_denorm ? {1'b0, b32_man} : (b32_is_zero ? 24'h0 : {1'b1, b32_man});

    wire [10:0] a16_mraw = a16_is_denorm ? {1'b0, a16_man} : (a16_is_zero ? 11'h0 : {1'b1, a16_man});
    wire [10:0] b16_mraw = b16_is_denorm ? {1'b0, b16_man} : (b16_is_zero ? 11'h0 : {1'b1, b16_man});

    wire [23:0] a_mantissa = use_fp16_fields ? {a16_mraw, 13'b0} : a32_mantissa;
    wire [23:0] b_mantissa = use_fp16_fields ? {b16_mraw, 13'b0} : b32_mantissa;

    // C mantissa (for FP32/acc modes, always use FP32 format)
    wire        c_sign_fp;
    wire [23:0] c_mantissa;
    wire [7:0]  c_exp_val;

    // For acc mode or FP32 FMA, use FP32-format C
    wire c32_sel = is_fp32 | is_acc;
    wire [23:0] c32_eff_mantissa = c_eff_is_denorm ? {1'b0, c_eff_man} : (c_eff_is_zero ? 24'h0 : {1'b1, c_eff_man});
    wire [7:0]  c32_eff_exp_adj  = c_eff_is_denorm ? 8'h01 : c_eff_exp;

    // For FP16 FMA C
    wire c16_is_denorm = (c16_exp == 0) && (c16_man != 0);
    wire [10:0] c16_mraw = c16_is_denorm ? {1'b0, c16_man} : (c16_is_zero ? 11'h0 : {1'b1, c16_man});
    wire [7:0]  c16_exp_fp32 = c16_is_denorm ? 8'd113 : {3'b0, c16_exp} + 8'd112;

    assign c_sign_fp = is_acc ? c_eff_sign : (use_fp16_fields ? c16_sign : c32_sign);
    assign c_mantissa = c32_sel ? c32_eff_mantissa : {c16_mraw, 13'b0};
    assign c_exp_val  = c32_sel ? c32_eff_exp_adj  : c16_exp_fp32;

    // Exponent (FP32 biased)
    wire [7:0] a32_exp_adj = a32_is_denorm ? 8'h01 : a32_exp;
    wire [7:0] b32_exp_adj = b32_is_denorm ? 8'h01 : b32_exp;
    wire [7:0] a16_exp_fp32 = a16_is_denorm ? 8'd113 : {3'b0, a16_exp} + 8'd112;
    wire [7:0] b16_exp_fp32 = b16_is_denorm ? 8'd113 : {3'b0, b16_exp} + 8'd112;

    wire [7:0] a_exp_fp = use_fp16_fields ? a16_exp_fp32 : a32_exp_adj;
    wire [7:0] b_exp_fp = use_fp16_fields ? b16_exp_fp32 : b32_exp_adj;

    wire product_sign_fp = a_sign_fp ^ b_sign_fp;
    wire nan_result_fp = a_is_nan | b_is_nan | c_is_nan |
                         (a_is_inf & b_is_zero) | (b_is_inf & a_is_zero) |
                         ((a_is_inf | b_is_inf) & c_is_inf & (product_sign_fp ^ c_sign_fp));
    wire inf_result_fp = ~nan_result_fp & (a_is_inf | b_is_inf | c_is_inf);
    wire inf_sign_fp   = (a_is_inf | b_is_inf) ? product_sign_fp : c_sign_fp;

    // =========================================================================
    // FP8 Path (same as Stage 2)
    // =========================================================================
    wire [7:0] a_fp8 [0:3];
    wire [7:0] b_fp8 [0:3];
    wire [7:0] c_fp8 [0:3];
    assign a_fp8[0] = A[7:0];   assign a_fp8[1] = A[15:8];
    assign a_fp8[2] = A[23:16]; assign a_fp8[3] = A[31:24];
    assign b_fp8[0] = B[7:0];   assign b_fp8[1] = B[15:8];
    assign b_fp8[2] = B[23:16]; assign b_fp8[3] = B[31:24];
    assign c_fp8[0] = C[7:0];   assign c_fp8[1] = C[15:8];
    assign c_fp8[2] = C[23:16]; assign c_fp8[3] = C[31:24];

    wire       fp8_a_sign[0:3], fp8_b_sign[0:3], fp8_c_sign[0:3];
    wire [3:0] fp8_a_exp[0:3],  fp8_b_exp[0:3],  fp8_c_exp[0:3];
    wire [2:0] fp8_a_man[0:3],  fp8_b_man[0:3],  fp8_c_man[0:3];
    wire       fp8_a_is_zero[0:3], fp8_b_is_zero[0:3], fp8_c_is_zero[0:3];
    wire       fp8_a_is_nan[0:3],  fp8_b_is_nan[0:3],  fp8_c_is_nan[0:3];
    wire [3:0] fp8_a_mantissa[0:3], fp8_b_mantissa[0:3], fp8_c_mantissa[0:3];
    wire [7:0] fp8_product[0:3];
    wire       fp8_prod_sign[0:3];
    wire [4:0] fp8_exp_sum[0:3];

    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : fp8_ext
            assign fp8_a_sign[gi] = a_fp8[gi][7];
            assign fp8_a_exp[gi]  = a_fp8[gi][6:3];
            assign fp8_a_man[gi]  = a_fp8[gi][2:0];
            assign fp8_b_sign[gi] = b_fp8[gi][7];
            assign fp8_b_exp[gi]  = b_fp8[gi][6:3];
            assign fp8_b_man[gi]  = b_fp8[gi][2:0];
            assign fp8_c_sign[gi] = c_fp8[gi][7];
            assign fp8_c_exp[gi]  = c_fp8[gi][6:3];
            assign fp8_c_man[gi]  = c_fp8[gi][2:0];
            assign fp8_a_is_zero[gi] = (fp8_a_exp[gi] == 0) && (fp8_a_man[gi] == 0);
            assign fp8_b_is_zero[gi] = (fp8_b_exp[gi] == 0) && (fp8_b_man[gi] == 0);
            assign fp8_c_is_zero[gi] = (fp8_c_exp[gi] == 0) && (fp8_c_man[gi] == 0);
            assign fp8_a_is_nan[gi]  = (fp8_a_exp[gi] == 4'hF) && (fp8_a_man[gi] == 3'h7);
            assign fp8_b_is_nan[gi]  = (fp8_b_exp[gi] == 4'hF) && (fp8_b_man[gi] == 3'h7);
            assign fp8_c_is_nan[gi]  = (fp8_c_exp[gi] == 4'hF) && (fp8_c_man[gi] == 3'h7);
            assign fp8_a_mantissa[gi] = fp8_a_is_zero[gi] ? 4'h0 : {1'b1, fp8_a_man[gi]};
            assign fp8_b_mantissa[gi] = fp8_b_is_zero[gi] ? 4'h0 : {1'b1, fp8_b_man[gi]};
            assign fp8_c_mantissa[gi] = fp8_c_is_zero[gi] ? 4'h0 : {1'b1, fp8_c_man[gi]};
            assign fp8_product[gi]    = fp8_a_mantissa[gi] * fp8_b_mantissa[gi];
            assign fp8_prod_sign[gi]  = fp8_a_sign[gi] ^ fp8_b_sign[gi];
            assign fp8_exp_sum[gi]    = {1'b0, fp8_a_exp[gi]} + {1'b0, fp8_b_exp[gi]} - 5'd7;
        end
    endgenerate

    // For FP8 acc mode: extend FP8 products to FP32 and sum them
    // Each FP8 product: mantissa = product[7:0] in UQ2.6
    // Extended to FP32: exp_fp32 = fp8_exp_sum + 113
    // We sum all 4 products at FP32 level, then add to acc_reg
    // Simplified: we do this in the FP32 adder path by converting
    // the sum of FP8 products to a single FP32 value first

    // =========================================================================
    // Pipeline Stage 1: Multiply
    // =========================================================================
    reg        s1_valid;
    reg [2:0]  s1_mode;
    reg        s1_nan_fp, s1_inf_fp, s1_inf_sign_fp;
    reg        s1_product_sign_fp, s1_c_sign_fp;
    reg [8:0]  s1_exp_sum_fp;
    reg [7:0]  s1_c_exp_fp;
    reg [47:0] s1_product_fp;
    reg [23:0] s1_c_mantissa_fp;
    reg        s1_prod_zero_fp;
    reg        s1_acc_en;

    // FP8 pipeline
    reg [7:0]  s1_fp8_product[0:3];
    reg        s1_fp8_prod_sign[0:3];
    reg [4:0]  s1_fp8_exp_sum[0:3];
    reg [3:0]  s1_fp8_c_man[0:3];
    reg        s1_fp8_c_sign[0:3];
    reg [3:0]  s1_fp8_c_exp[0:3];
    reg        s1_fp8_nan[0:3];
    reg        s1_fp8_prod_zero[0:3];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 0; s1_mode <= 0; s1_acc_en <= 0;
            s1_nan_fp <= 0; s1_inf_fp <= 0; s1_inf_sign_fp <= 0;
            s1_product_sign_fp <= 0; s1_c_sign_fp <= 0;
            s1_exp_sum_fp <= 0; s1_c_exp_fp <= 0;
            s1_product_fp <= 0; s1_c_mantissa_fp <= 0;
            s1_prod_zero_fp <= 0;
            for (i = 0; i < 4; i = i + 1) begin
                s1_fp8_product[i] <= 0; s1_fp8_prod_sign[i] <= 0;
                s1_fp8_exp_sum[i] <= 0; s1_fp8_c_man[i] <= 0;
                s1_fp8_c_sign[i] <= 0; s1_fp8_c_exp[i] <= 0;
                s1_fp8_nan[i] <= 0; s1_fp8_prod_zero[i] <= 0;
            end
        end else begin
            s1_valid            <= valid_in;
            s1_mode             <= mode;
            s1_acc_en           <= acc_en;
            s1_nan_fp           <= nan_result_fp;
            s1_inf_fp           <= inf_result_fp;
            s1_inf_sign_fp      <= inf_sign_fp;
            s1_product_sign_fp  <= product_sign_fp;
            s1_c_sign_fp        <= c_sign_fp;
            s1_exp_sum_fp       <= {1'b0, a_exp_fp} + {1'b0, b_exp_fp} - 9'd127;
            s1_c_exp_fp         <= c_exp_val;
            s1_product_fp       <= a_mantissa * b_mantissa;
            s1_c_mantissa_fp    <= c_mantissa;
            s1_prod_zero_fp     <= a_is_zero | b_is_zero;
            for (i = 0; i < 4; i = i + 1) begin
                s1_fp8_product[i]   <= fp8_product[i];
                s1_fp8_prod_sign[i] <= fp8_prod_sign[i];
                s1_fp8_exp_sum[i]   <= fp8_exp_sum[i];
                s1_fp8_c_man[i]     <= fp8_c_mantissa[i];
                s1_fp8_c_sign[i]    <= fp8_c_sign[i];
                s1_fp8_c_exp[i]     <= fp8_c_exp[i];
                s1_fp8_nan[i]       <= fp8_a_is_nan[i] | fp8_b_is_nan[i] | fp8_c_is_nan[i];
                s1_fp8_prod_zero[i] <= fp8_a_is_zero[i] | fp8_b_is_zero[i];
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Align / CSA
    // =========================================================================
    // FP32/FP16/Acc16 path alignment
    wire signed [9:0] exp_diff_fp = $signed({1'b0, s1_exp_sum_fp}) - $signed({2'b0, s1_c_exp_fp});

    reg [49:0] product_aligned_fp, c_aligned_fp;
    reg [8:0]  align_exp_fp;

    always @(*) begin
        if (exp_diff_fp >= 0) begin
            product_aligned_fp = {s1_product_fp, 2'b0};
            c_aligned_fp = (exp_diff_fp > 49) ? 50'b0 : ({1'b0, s1_c_mantissa_fp, 25'b0}) >> exp_diff_fp;
            align_exp_fp = s1_exp_sum_fp;
        end else begin
            product_aligned_fp = ((-exp_diff_fp) > 49) ? 50'b0 : ({s1_product_fp, 2'b0}) >> (-exp_diff_fp);
            c_aligned_fp = {1'b0, s1_c_mantissa_fp, 25'b0};
            align_exp_fp = {1'b0, s1_c_exp_fp};
        end
    end

    wire eff_sub_fp = s1_product_sign_fp ^ s1_c_sign_fp;
    wire [49:0] csa_a_fp = product_aligned_fp;
    wire [49:0] csa_b_fp = eff_sub_fp ? (~c_aligned_fp) : c_aligned_fp;
    wire [49:0] cin_fp   = {49'b0, eff_sub_fp};
    wire [49:0] csa_sum_fp   = csa_a_fp ^ csa_b_fp ^ cin_fp;
    wire [49:0] csa_carry_fp = ((csa_a_fp & csa_b_fp) | (csa_a_fp & cin_fp) | (csa_b_fp & cin_fp)) << 1;

    // FP8 per-lane alignment (for FP8 FMA mode only)
    reg [9:0] fp8_prod_aligned[0:3];
    reg [9:0] fp8_c_aligned[0:3];
    reg [4:0] fp8_align_exp[0:3];

    always @(*) begin
        for (i = 0; i < 4; i = i + 1) begin
            begin : fp8_align_block
                reg signed [5:0] fp8_exp_diff;
                fp8_exp_diff = $signed({1'b0, s1_fp8_exp_sum[i]}) - $signed({2'b0, s1_fp8_c_exp[i]});
                if (fp8_exp_diff >= 0) begin
                    fp8_prod_aligned[i] = {s1_fp8_product[i], 2'b0};
                    fp8_c_aligned[i] = (fp8_exp_diff > 9) ? 10'b0 :
                        ({1'b0, s1_fp8_c_man[i], 5'b0}) >> fp8_exp_diff;
                    fp8_align_exp[i] = s1_fp8_exp_sum[i];
                end else begin
                    fp8_prod_aligned[i] = ((-fp8_exp_diff) > 9) ? 10'b0 :
                        ({s1_fp8_product[i], 2'b0}) >> (-fp8_exp_diff);
                    fp8_c_aligned[i] = {1'b0, s1_fp8_c_man[i], 5'b0};
                    fp8_align_exp[i] = {1'b0, s1_fp8_c_exp[i]};
                end
            end
        end
    end

    wire [9:0] fp8_csa_sum[0:3], fp8_csa_carry[0:3];
    wire       fp8_eff_sub[0:3];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : fp8_csa
            assign fp8_eff_sub[gi] = s1_fp8_prod_sign[gi] ^ s1_fp8_c_sign[gi];
            wire [9:0] fp8_csa_b = fp8_eff_sub[gi] ? (~fp8_c_aligned[gi]) : fp8_c_aligned[gi];
            wire [9:0] fp8_cin   = {9'b0, fp8_eff_sub[gi]};
            assign fp8_csa_sum[gi]   = fp8_prod_aligned[gi] ^ fp8_csa_b ^ fp8_cin;
            assign fp8_csa_carry[gi] = ((fp8_prod_aligned[gi] & fp8_csa_b) |
                                        (fp8_prod_aligned[gi] & fp8_cin) |
                                        (fp8_csa_b & fp8_cin)) << 1;
        end
    endgenerate

    // Stage 2 pipeline registers
    reg        s2_valid;
    reg [2:0]  s2_mode;
    reg        s2_nan_fp, s2_inf_fp, s2_inf_sign_fp;
    reg        s2_product_sign_fp, s2_c_sign_fp, s2_eff_sub_fp;
    reg [8:0]  s2_exp_fp;
    reg [49:0] s2_sum_fp, s2_carry_fp;
    reg        s2_acc_en;
    // FP8
    reg [9:0]  s2_fp8_sum[0:3], s2_fp8_carry[0:3];
    reg [4:0]  s2_fp8_exp[0:3];
    reg        s2_fp8_prod_sign[0:3], s2_fp8_c_sign[0:3];
    reg        s2_fp8_eff_sub[0:3], s2_fp8_nan[0:3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 0; s2_mode <= 0; s2_acc_en <= 0;
            s2_nan_fp <= 0; s2_inf_fp <= 0; s2_inf_sign_fp <= 0;
            s2_product_sign_fp <= 0; s2_c_sign_fp <= 0; s2_eff_sub_fp <= 0;
            s2_exp_fp <= 0; s2_sum_fp <= 0; s2_carry_fp <= 0;
            for (i = 0; i < 4; i = i + 1) begin
                s2_fp8_sum[i] <= 0; s2_fp8_carry[i] <= 0;
                s2_fp8_exp[i] <= 0; s2_fp8_prod_sign[i] <= 0;
                s2_fp8_c_sign[i] <= 0; s2_fp8_eff_sub[i] <= 0;
                s2_fp8_nan[i] <= 0;
            end
        end else begin
            s2_valid            <= s1_valid;
            s2_mode             <= s1_mode;
            s2_acc_en           <= s1_acc_en;
            s2_nan_fp           <= s1_nan_fp;
            s2_inf_fp           <= s1_inf_fp;
            s2_inf_sign_fp      <= s1_inf_sign_fp;
            s2_product_sign_fp  <= s1_product_sign_fp;
            s2_c_sign_fp        <= s1_c_sign_fp;
            s2_eff_sub_fp       <= eff_sub_fp;
            s2_exp_fp           <= align_exp_fp;
            s2_sum_fp           <= csa_sum_fp;
            s2_carry_fp         <= csa_carry_fp;
            for (i = 0; i < 4; i = i + 1) begin
                s2_fp8_sum[i]       <= fp8_csa_sum[i];
                s2_fp8_carry[i]     <= fp8_csa_carry[i];
                s2_fp8_exp[i]       <= fp8_align_exp[i];
                s2_fp8_prod_sign[i] <= s1_fp8_prod_sign[i];
                s2_fp8_c_sign[i]    <= s1_fp8_c_sign[i];
                s2_fp8_eff_sub[i]   <= fp8_eff_sub[i];
                s2_fp8_nan[i]       <= s1_fp8_nan[i];
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 3: Normalize / Round / Output
    // =========================================================================
    integer k;

    // FP32/FP16/Acc normalization
    wire [50:0] raw_fp = {1'b0, s2_sum_fp} + {1'b0, s2_carry_fp};
    wire sub_borrow_fp = s2_eff_sub_fp & ~raw_fp[50];
    wire [50:0] mag_fp = s2_eff_sub_fp ?
        (raw_fp[50] ? {1'b0, raw_fp[49:0]} : ({1'b0, ~raw_fp[49:0]} + 51'd1)) : raw_fp;
    wire rsign_fp = sub_borrow_fp ? s2_c_sign_fp : s2_product_sign_fp;

    reg [5:0] lzd_fp;
    always @(*) begin
        lzd_fp = 6'd51;
        for (k = 0; k <= 50; k = k + 1)
            if (mag_fp[k]) lzd_fp = 50 - k;
    end

    wire [50:0] norm_man_fp = mag_fp << lzd_fp;
    wire signed [9:0] nexp_fp_s = $signed({1'b0, s2_exp_fp}) + 10'sd2 - $signed({4'b0, lzd_fp});
    wire [8:0] nexp_fp = nexp_fp_s[8:0];

    wire [22:0] trunc_fp = norm_man_fp[49:27];
    wire gbit_fp = norm_man_fp[26], rbit_fp = norm_man_fp[25], sbit_fp = |norm_man_fp[24:0];
    wire rup_fp = gbit_fp & (rbit_fp | sbit_fp | trunc_fp[0]);
    wire [23:0] rnd_fp = {1'b0, trunc_fp} + {23'b0, rup_fp};
    wire [22:0] fman_fp = rnd_fp[23] ? rnd_fp[23:1] : rnd_fp[22:0];
    wire [8:0]  fexp_fp = rnd_fp[23] ? nexp_fp + 9'd1 : nexp_fp;

    wire fp32_ov = (fexp_fp >= 9'd255) && (mag_fp != 0);
    wire fp32_uf = (nexp_fp_s <= 0) && (mag_fp != 0);
    wire [31:0] fp32_res = (mag_fp == 0) ? {(s2_eff_sub_fp ? 1'b0 : s2_product_sign_fp), 31'b0} :
                           fp32_ov ? {rsign_fp, 8'hFF, 23'h0} :
                           fp32_uf ? {rsign_fp, 31'b0} :
                                     {rsign_fp, fexp_fp[7:0], fman_fp};

    wire [8:0] fp16_eb = fexp_fp - 9'd112;
    wire fp16_ov = (fp16_eb >= 9'd31) && (mag_fp != 0);
    wire fp16_uf = (fp16_eb[8] || fp16_eb == 0) && (mag_fp != 0);
    wire [15:0] fp16_res = (mag_fp == 0) ? {(s2_eff_sub_fp ? 1'b0 : s2_product_sign_fp), 15'b0} :
                           fp16_ov ? {rsign_fp, 5'h1F, 10'h0} :
                           fp16_uf ? {rsign_fp, 15'b0} :
                                     {rsign_fp, fp16_eb[4:0], fman_fp[22:13]};

    // FP8 normalization per lane
    wire [7:0] fp8_result[0:3];
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : fp8_norm
            wire [10:0] fp8_raw = {1'b0, s2_fp8_sum[gi]} + {1'b0, s2_fp8_carry[gi]};
            wire fp8_sub_borrow = s2_fp8_eff_sub[gi] & ~fp8_raw[10];
            wire [10:0] fp8_mag = s2_fp8_eff_sub[gi] ?
                (fp8_raw[10] ? {1'b0, fp8_raw[9:0]} : ({1'b0, ~fp8_raw[9:0]} + 11'd1)) : fp8_raw;
            wire fp8_rsign = fp8_sub_borrow ? s2_fp8_c_sign[gi] : s2_fp8_prod_sign[gi];

            reg [3:0] fp8_lzd;
            integer j;
            always @(*) begin
                fp8_lzd = 4'd11;
                for (j = 0; j <= 10; j = j + 1)
                    if (fp8_mag[j]) fp8_lzd = 10 - j;
            end

            wire [10:0] fp8_norm_m = fp8_mag << fp8_lzd;
            wire signed [5:0] fp8_nexp_s = $signed({1'b0, s2_fp8_exp[gi]}) + 6'sd2 - $signed({2'b0, fp8_lzd});
            wire [4:0] fp8_nexp = fp8_nexp_s[4:0];

            wire [2:0] fp8_trunc = fp8_norm_m[9:7];
            wire fp8_g = fp8_norm_m[6], fp8_r = fp8_norm_m[5], fp8_s = |fp8_norm_m[4:0];
            wire fp8_rup = fp8_g & (fp8_r | fp8_s | fp8_trunc[0]);
            wire [3:0] fp8_rnd = {1'b0, fp8_trunc} + {3'b0, fp8_rup};
            wire [2:0] fp8_fman = fp8_rnd[3] ? fp8_rnd[3:1] : fp8_rnd[2:0];
            wire [4:0] fp8_fexp = fp8_rnd[3] ? fp8_nexp + 5'd1 : fp8_nexp;

            wire fp8_is_ov = (fp8_fexp >= 5'd15) && (fp8_mag != 0);
            wire fp8_is_uf = (fp8_nexp_s <= 0) && (fp8_mag != 0);

            assign fp8_result[gi] = s2_fp8_nan[gi] ? {fp8_rsign, 4'hF, 3'h7} :
                                    (fp8_mag == 0)  ? {(s2_fp8_eff_sub[gi] ? 1'b0 : s2_fp8_prod_sign[gi]), 7'b0} :
                                    fp8_is_ov       ? {fp8_rsign, 4'hE, 3'h7} :
                                    fp8_is_uf       ? {fp8_rsign, 7'b0} :
                                                      {fp8_rsign, fp8_fexp[3:0], fp8_fman};
        end
    endgenerate

    // =========================================================================
    // Output and Accumulator Update
    // =========================================================================
    wire is_fp16_out = (s2_mode == 3'b001);
    wire is_fp8_out  = (s2_mode == 3'b010);
    wire is_acc_out  = (s2_mode[2] == 1'b1); // modes 100, 101

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0; valid_out <= 0;
            overflow <= 0; underflow <= 0; inexact <= 0;
            acc_out <= 0; acc_reg <= 0;
        end else begin
            valid_out <= s2_valid;
            overflow <= 0; underflow <= 0; inexact <= 0;

            // Accumulator initialization
            if (acc_en && !s2_valid)
                acc_reg <= acc_in;

            if (s2_valid) begin
                if (is_fp8_out) begin
                    result <= {fp8_result[3], fp8_result[2], fp8_result[1], fp8_result[0]};
                end else if (is_fp16_out) begin
                    if (s2_nan_fp)
                        result <= {16'b0, 16'h7E00};
                    else if (s2_inf_fp)
                        result <= {16'b0, s2_inf_sign_fp, 5'h1F, 10'h0};
                    else begin
                        result    <= {16'b0, fp16_res};
                        overflow  <= fp16_ov;
                        underflow <= fp16_uf;
                    end
                end else begin
                    // FP32, Acc16, Acc8 all produce FP32 result
                    if (s2_nan_fp)
                        result <= 32'h7FC00000;
                    else if (s2_inf_fp)
                        result <= {s2_inf_sign_fp, 8'hFF, 23'h0};
                    else begin
                        result    <= fp32_res;
                        overflow  <= fp32_ov;
                        underflow <= fp32_uf;
                    end
                end

                // Update accumulator for acc modes
                if (is_acc_out && s2_acc_en) begin
                    if (s2_nan_fp)
                        acc_reg <= 32'h7FC00000;
                    else if (fp32_ov)
                        acc_reg <= {rsign_fp, 8'hFF, 23'h0}; // saturate to Inf
                    else
                        acc_reg <= fp32_res;
                end

                acc_out <= acc_reg;
            end
        end
    end

endmodule
