// =============================================================================
// Stage 2: FP32 + FP16 + FP8x4 FMA (Fused Multiply-Add)
// Supports: mode 00=FP32, mode 01=FP16, mode 10=FP8x4
// Incremental on Stage 1: adds four parallel FP8 (E4M3) lanes
// 3-stage pipeline: Stage1=Multiply, Stage2=Align/CSA, Stage3=Normalize/Round
// =============================================================================

module fma_fp8x4_stage2 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [1:0]  mode,       // 00=FP32, 01=FP16, 10=FP8x4
    input  wire [31:0] A,
    input  wire [31:0] B,
    input  wire [31:0] C,
    output reg  [31:0] result,
    output reg         valid_out,
    output reg         overflow,
    output reg         underflow,
    output reg         inexact
);

    wire is_fp16 = (mode == 2'b01);
    wire is_fp8  = (mode == 2'b10);

    // =========================================================================
    // FP32/FP16 Path (reuse from Stage 1)
    // =========================================================================
    // FP32 fields
    wire        a32_sign = A[31]; wire [7:0] a32_exp = A[30:23]; wire [22:0] a32_man = A[22:0];
    wire        b32_sign = B[31]; wire [7:0] b32_exp = B[30:23]; wire [22:0] b32_man = B[22:0];
    wire        c32_sign = C[31]; wire [7:0] c32_exp = C[30:23]; wire [22:0] c32_man = C[22:0];

    // FP16 fields
    wire        a16_sign = A[15]; wire [4:0] a16_exp = A[14:10]; wire [9:0] a16_man = A[9:0];
    wire        b16_sign = B[15]; wire [4:0] b16_exp = B[14:10]; wire [9:0] b16_man = B[9:0];
    wire        c16_sign = C[15]; wire [4:0] c16_exp = C[14:10]; wire [9:0] c16_man = C[9:0];

    // Special flags - FP32
    wire a32_is_zero = (a32_exp == 8'h00) && (a32_man == 23'h0);
    wire b32_is_zero = (b32_exp == 8'h00) && (b32_man == 23'h0);
    wire c32_is_zero = (c32_exp == 8'h00) && (c32_man == 23'h0);
    wire a32_is_inf  = (a32_exp == 8'hFF) && (a32_man == 23'h0);
    wire b32_is_inf  = (b32_exp == 8'hFF) && (b32_man == 23'h0);
    wire c32_is_inf  = (c32_exp == 8'hFF) && (c32_man == 23'h0);
    wire a32_is_nan  = (a32_exp == 8'hFF) && (a32_man != 23'h0);
    wire b32_is_nan  = (b32_exp == 8'hFF) && (b32_man != 23'h0);
    wire c32_is_nan  = (c32_exp == 8'hFF) && (c32_man != 23'h0);
    wire a32_is_denorm = (a32_exp == 8'h00) && (a32_man != 23'h0);
    wire b32_is_denorm = (b32_exp == 8'h00) && (b32_man != 23'h0);
    wire c32_is_denorm = (c32_exp == 8'h00) && (c32_man != 23'h0);

    // Special flags - FP16
    wire a16_is_zero = (a16_exp == 5'h00) && (a16_man == 10'h0);
    wire b16_is_zero = (b16_exp == 5'h00) && (b16_man == 10'h0);
    wire c16_is_zero = (c16_exp == 5'h00) && (c16_man == 10'h0);
    wire a16_is_inf  = (a16_exp == 5'h1F) && (a16_man == 10'h0);
    wire b16_is_inf  = (b16_exp == 5'h1F) && (b16_man == 10'h0);
    wire c16_is_inf  = (c16_exp == 5'h1F) && (c16_man == 10'h0);
    wire a16_is_nan  = (a16_exp == 5'h1F) && (a16_man != 10'h0);
    wire b16_is_nan  = (b16_exp == 5'h1F) && (b16_man != 10'h0);
    wire c16_is_nan  = (c16_exp == 5'h1F) && (c16_man != 10'h0);
    wire a16_is_denorm = (a16_exp == 5'h00) && (a16_man != 10'h0);
    wire b16_is_denorm = (b16_exp == 5'h00) && (b16_man != 10'h0);
    wire c16_is_denorm = (c16_exp == 5'h00) && (c16_man != 10'h0);

    // Unified flags for FP32/FP16
    wire a_sign_fp  = is_fp16 ? a16_sign : a32_sign;
    wire b_sign_fp  = is_fp16 ? b16_sign : b32_sign;
    wire c_sign_fp  = is_fp16 ? c16_sign : c32_sign;
    wire a_is_zero  = is_fp16 ? a16_is_zero : a32_is_zero;
    wire b_is_zero  = is_fp16 ? b16_is_zero : b32_is_zero;
    wire c_is_zero  = is_fp16 ? c16_is_zero : c32_is_zero;
    wire a_is_inf   = is_fp16 ? a16_is_inf  : a32_is_inf;
    wire b_is_inf   = is_fp16 ? b16_is_inf  : b32_is_inf;
    wire c_is_inf   = is_fp16 ? c16_is_inf  : c32_is_inf;
    wire a_is_nan   = is_fp16 ? a16_is_nan  : a32_is_nan;
    wire b_is_nan   = is_fp16 ? b16_is_nan  : b32_is_nan;
    wire c_is_nan   = is_fp16 ? c16_is_nan  : c32_is_nan;

    // Unified mantissa (24-bit FP32 format)
    wire [23:0] a32_mantissa = a32_is_denorm ? {1'b0, a32_man} : (a32_is_zero ? 24'h0 : {1'b1, a32_man});
    wire [23:0] b32_mantissa = b32_is_denorm ? {1'b0, b32_man} : (b32_is_zero ? 24'h0 : {1'b1, b32_man});
    wire [23:0] c32_mantissa = c32_is_denorm ? {1'b0, c32_man} : (c32_is_zero ? 24'h0 : {1'b1, c32_man});

    wire [10:0] a16_mraw = a16_is_denorm ? {1'b0, a16_man} : (a16_is_zero ? 11'h0 : {1'b1, a16_man});
    wire [10:0] b16_mraw = b16_is_denorm ? {1'b0, b16_man} : (b16_is_zero ? 11'h0 : {1'b1, b16_man});
    wire [10:0] c16_mraw = c16_is_denorm ? {1'b0, c16_man} : (c16_is_zero ? 11'h0 : {1'b1, c16_man});
    wire [23:0] a16_mantissa = {a16_mraw, 13'b0};
    wire [23:0] b16_mantissa = {b16_mraw, 13'b0};
    wire [23:0] c16_mantissa = {c16_mraw, 13'b0};

    wire [23:0] a_mantissa = is_fp16 ? a16_mantissa : a32_mantissa;
    wire [23:0] b_mantissa = is_fp16 ? b16_mantissa : b32_mantissa;
    wire [23:0] c_mantissa = is_fp16 ? c16_mantissa : c32_mantissa;

    // Unified exponent (FP32 biased)
    wire [7:0] a32_exp_adj = a32_is_denorm ? 8'h01 : a32_exp;
    wire [7:0] b32_exp_adj = b32_is_denorm ? 8'h01 : b32_exp;
    wire [7:0] c32_exp_adj = c32_is_denorm ? 8'h01 : c32_exp;
    wire [7:0] a16_exp_fp32 = a16_is_denorm ? 8'd113 : {3'b0, a16_exp} + 8'd112;
    wire [7:0] b16_exp_fp32 = b16_is_denorm ? 8'd113 : {3'b0, b16_exp} + 8'd112;
    wire [7:0] c16_exp_fp32 = c16_is_denorm ? 8'd113 : {3'b0, c16_exp} + 8'd112;

    wire [7:0] a_exp_fp = is_fp16 ? a16_exp_fp32 : a32_exp_adj;
    wire [7:0] b_exp_fp = is_fp16 ? b16_exp_fp32 : b32_exp_adj;
    wire [7:0] c_exp_fp = is_fp16 ? c16_exp_fp32 : c32_exp_adj;

    wire product_sign_fp = a_sign_fp ^ b_sign_fp;
    wire nan_result_fp = a_is_nan | b_is_nan | c_is_nan |
                         (a_is_inf & b_is_zero) | (b_is_inf & a_is_zero) |
                         ((a_is_inf | b_is_inf) & c_is_inf & (product_sign_fp ^ c_sign_fp));
    wire inf_result_fp = ~nan_result_fp & (a_is_inf | b_is_inf | c_is_inf);
    wire inf_sign_fp   = (a_is_inf | b_is_inf) ? product_sign_fp : c_sign_fp;

    // =========================================================================
    // FP8 x4 Path - Four independent lanes
    // =========================================================================
    // FP8 E4M3 format: [7] sign, [6:3] exp (bias=7), [2:0] mantissa
    wire [7:0] a_fp8 [0:3];
    wire [7:0] b_fp8 [0:3];
    wire [7:0] c_fp8 [0:3];

    assign a_fp8[0] = A[7:0];   assign a_fp8[1] = A[15:8];
    assign a_fp8[2] = A[23:16]; assign a_fp8[3] = A[31:24];
    assign b_fp8[0] = B[7:0];   assign b_fp8[1] = B[15:8];
    assign b_fp8[2] = B[23:16]; assign b_fp8[3] = B[31:24];
    assign c_fp8[0] = C[7:0];   assign c_fp8[1] = C[15:8];
    assign c_fp8[2] = C[23:16]; assign c_fp8[3] = C[31:24];

    // Per-lane field extraction
    wire       fp8_a_sign [0:3];
    wire [3:0] fp8_a_exp  [0:3];
    wire [2:0] fp8_a_man  [0:3];
    wire       fp8_b_sign [0:3];
    wire [3:0] fp8_b_exp  [0:3];
    wire [2:0] fp8_b_man  [0:3];
    wire       fp8_c_sign [0:3];
    wire [3:0] fp8_c_exp  [0:3];
    wire [2:0] fp8_c_man  [0:3];

    wire       fp8_a_is_zero [0:3];
    wire       fp8_b_is_zero [0:3];
    wire       fp8_c_is_zero [0:3];
    wire       fp8_a_is_nan  [0:3];
    wire       fp8_b_is_nan  [0:3];
    wire       fp8_c_is_nan  [0:3];

    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : fp8_extract
            assign fp8_a_sign[gi] = a_fp8[gi][7];
            assign fp8_a_exp[gi]  = a_fp8[gi][6:3];
            assign fp8_a_man[gi]  = a_fp8[gi][2:0];
            assign fp8_b_sign[gi] = b_fp8[gi][7];
            assign fp8_b_exp[gi]  = b_fp8[gi][6:3];
            assign fp8_b_man[gi]  = b_fp8[gi][2:0];
            assign fp8_c_sign[gi] = c_fp8[gi][7];
            assign fp8_c_exp[gi]  = c_fp8[gi][6:3];
            assign fp8_c_man[gi]  = c_fp8[gi][2:0];

            // E4M3: NaN when exp=15 and man=7 (max values)
            assign fp8_a_is_zero[gi] = (fp8_a_exp[gi] == 4'h0) && (fp8_a_man[gi] == 3'h0);
            assign fp8_b_is_zero[gi] = (fp8_b_exp[gi] == 4'h0) && (fp8_b_man[gi] == 3'h0);
            assign fp8_c_is_zero[gi] = (fp8_c_exp[gi] == 4'h0) && (fp8_c_man[gi] == 3'h0);
            assign fp8_a_is_nan[gi]  = (fp8_a_exp[gi] == 4'hF) && (fp8_a_man[gi] == 3'h7);
            assign fp8_b_is_nan[gi]  = (fp8_b_exp[gi] == 4'hF) && (fp8_b_man[gi] == 3'h7);
            assign fp8_c_is_nan[gi]  = (fp8_c_exp[gi] == 4'hF) && (fp8_c_man[gi] == 3'h7);
        end
    endgenerate

    // Per-lane FP8 FMA computation (combinational, registered at pipeline stages)
    // Mantissa: {1, man[2:0]} = 4 bits, product = 4x4 = 8 bits (UQ2.6)
    wire [3:0]  fp8_a_mantissa [0:3];
    wire [3:0]  fp8_b_mantissa [0:3];
    wire [3:0]  fp8_c_mantissa [0:3];
    wire [7:0]  fp8_product    [0:3];  // 8-bit product (UQ2.6)
    wire        fp8_prod_sign  [0:3];
    wire [4:0]  fp8_exp_sum    [0:3];  // 5-bit to handle overflow

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : fp8_multiply
            assign fp8_a_mantissa[gi] = fp8_a_is_zero[gi] ? 4'h0 : {1'b1, fp8_a_man[gi]};
            assign fp8_b_mantissa[gi] = fp8_b_is_zero[gi] ? 4'h0 : {1'b1, fp8_b_man[gi]};
            assign fp8_c_mantissa[gi] = fp8_c_is_zero[gi] ? 4'h0 : {1'b1, fp8_c_man[gi]};
            assign fp8_product[gi]    = fp8_a_mantissa[gi] * fp8_b_mantissa[gi];
            assign fp8_prod_sign[gi]  = fp8_a_sign[gi] ^ fp8_b_sign[gi];
            assign fp8_exp_sum[gi]    = {1'b0, fp8_a_exp[gi]} + {1'b0, fp8_b_exp[gi]} - 5'd7;
        end
    endgenerate

    // =========================================================================
    // Pipeline Stage 1: Multiply (registered)
    // =========================================================================
    // FP32/FP16 path
    reg        s1_valid;
    reg [1:0]  s1_mode;
    reg        s1_nan_fp, s1_inf_fp, s1_inf_sign_fp;
    reg        s1_product_sign_fp, s1_c_sign_fp;
    reg [8:0]  s1_exp_sum_fp;
    reg [7:0]  s1_c_exp_fp;
    reg [47:0] s1_product_fp;
    reg [23:0] s1_c_mantissa_fp;
    reg        s1_prod_zero_fp;

    // FP8 path (4 lanes)
    reg [7:0]  s1_fp8_product  [0:3];
    reg        s1_fp8_prod_sign[0:3];
    reg [4:0]  s1_fp8_exp_sum  [0:3];
    reg [3:0]  s1_fp8_c_man    [0:3];
    reg        s1_fp8_c_sign   [0:3];
    reg [3:0]  s1_fp8_c_exp    [0:3];
    reg        s1_fp8_nan      [0:3];
    reg        s1_fp8_prod_zero[0:3];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 0; s1_mode <= 0;
            s1_nan_fp <= 0; s1_inf_fp <= 0; s1_inf_sign_fp <= 0;
            s1_product_sign_fp <= 0; s1_c_sign_fp <= 0;
            s1_exp_sum_fp <= 0; s1_c_exp_fp <= 0;
            s1_product_fp <= 0; s1_c_mantissa_fp <= 0;
            s1_prod_zero_fp <= 0;
            for (i = 0; i < 4; i = i + 1) begin
                s1_fp8_product[i]   <= 0;
                s1_fp8_prod_sign[i] <= 0;
                s1_fp8_exp_sum[i]   <= 0;
                s1_fp8_c_man[i]     <= 0;
                s1_fp8_c_sign[i]    <= 0;
                s1_fp8_c_exp[i]     <= 0;
                s1_fp8_nan[i]       <= 0;
                s1_fp8_prod_zero[i] <= 0;
            end
        end else begin
            s1_valid            <= valid_in;
            s1_mode             <= mode;
            s1_nan_fp           <= nan_result_fp;
            s1_inf_fp           <= inf_result_fp;
            s1_inf_sign_fp      <= inf_sign_fp;
            s1_product_sign_fp  <= product_sign_fp;
            s1_c_sign_fp        <= c_sign_fp;
            s1_exp_sum_fp       <= {1'b0, a_exp_fp} + {1'b0, b_exp_fp} - 9'd127;
            s1_c_exp_fp         <= c_exp_fp;
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
    // FP32/FP16 alignment (same as Stage 1)
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

    // FP8 alignment per lane
    // FP8 product is 8 bits UQ2.6. Extended to 10 bits: {product, 2'b0} (UQ2.8)
    // C mantissa is 4 bits UQ1.3. Placed as {1'b0, c_man, 5'b0} (UQ1.8 in 10 bits)
    // Binary point between bits 8 and 7

    reg [9:0] fp8_prod_aligned [0:3];
    reg [9:0] fp8_c_aligned    [0:3];
    reg [4:0] fp8_align_exp    [0:3];

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

    // FP8 CSA per lane
    wire [9:0] fp8_csa_sum   [0:3];
    wire [9:0] fp8_csa_carry [0:3];
    wire       fp8_eff_sub   [0:3];

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
    reg [1:0]  s2_mode;
    // FP32/FP16
    reg        s2_nan_fp, s2_inf_fp, s2_inf_sign_fp;
    reg        s2_product_sign_fp, s2_c_sign_fp, s2_eff_sub_fp;
    reg [8:0]  s2_exp_fp;
    reg [49:0] s2_sum_fp, s2_carry_fp;
    // FP8
    reg [9:0]  s2_fp8_sum   [0:3];
    reg [9:0]  s2_fp8_carry [0:3];
    reg [4:0]  s2_fp8_exp   [0:3];
    reg        s2_fp8_prod_sign [0:3];
    reg        s2_fp8_c_sign    [0:3];
    reg        s2_fp8_eff_sub   [0:3];
    reg        s2_fp8_nan       [0:3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 0; s2_mode <= 0;
            s2_nan_fp <= 0; s2_inf_fp <= 0; s2_inf_sign_fp <= 0;
            s2_product_sign_fp <= 0; s2_c_sign_fp <= 0; s2_eff_sub_fp <= 0;
            s2_exp_fp <= 0; s2_sum_fp <= 0; s2_carry_fp <= 0;
            for (i = 0; i < 4; i = i + 1) begin
                s2_fp8_sum[i]       <= 0;
                s2_fp8_carry[i]     <= 0;
                s2_fp8_exp[i]       <= 0;
                s2_fp8_prod_sign[i] <= 0;
                s2_fp8_c_sign[i]    <= 0;
                s2_fp8_eff_sub[i]   <= 0;
                s2_fp8_nan[i]       <= 0;
            end
        end else begin
            s2_valid            <= s1_valid;
            s2_mode             <= s1_mode;
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
    // Pipeline Stage 3: Normalize / Round
    // =========================================================================

    // --- FP32/FP16 normalization (same as Stage 1) ---
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

    // --- FP8 normalization per lane ---
    wire [7:0] fp8_result [0:3];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : fp8_norm
            wire [10:0] fp8_raw = {1'b0, s2_fp8_sum[gi]} + {1'b0, s2_fp8_carry[gi]};
            wire fp8_sub_borrow = s2_fp8_eff_sub[gi] & ~fp8_raw[10];
            wire [10:0] fp8_mag = s2_fp8_eff_sub[gi] ?
                (fp8_raw[10] ? {1'b0, fp8_raw[9:0]} : ({1'b0, ~fp8_raw[9:0]} + 11'd1)) :
                fp8_raw;
            wire fp8_rsign = fp8_sub_borrow ? s2_fp8_c_sign[gi] : s2_fp8_prod_sign[gi];

            // LZD for 11-bit magnitude
            reg [3:0] fp8_lzd;
            integer j;
            always @(*) begin
                fp8_lzd = 4'd11;
                for (j = 0; j <= 10; j = j + 1)
                    if (fp8_mag[j]) fp8_lzd = 10 - j;
            end

            wire [10:0] fp8_norm = fp8_mag << fp8_lzd;
            // Exponent: align_exp + 2 - lzd (binary point between bits 8 and 7)
            wire signed [5:0] fp8_nexp_s = $signed({1'b0, s2_fp8_exp[gi]}) + 6'sd2 - $signed({2'b0, fp8_lzd});
            wire [4:0] fp8_nexp = fp8_nexp_s[4:0];

            // Round: mantissa[9:7] = 3 bits, guard = [6], round = [5], sticky = |[4:0]
            wire [2:0] fp8_trunc = fp8_norm[9:7];
            wire fp8_g = fp8_norm[6], fp8_r = fp8_norm[5], fp8_s = |fp8_norm[4:0];
            wire fp8_rup = fp8_g & (fp8_r | fp8_s | fp8_trunc[0]);
            wire [3:0] fp8_rnd = {1'b0, fp8_trunc} + {3'b0, fp8_rup};
            wire [2:0] fp8_fman = fp8_rnd[3] ? fp8_rnd[3:1] : fp8_rnd[2:0];
            wire [4:0] fp8_fexp = fp8_rnd[3] ? fp8_nexp + 5'd1 : fp8_nexp;

            wire fp8_is_ov = (fp8_fexp >= 5'd15) && (fp8_mag != 0);
            wire fp8_is_uf = (fp8_nexp_s <= 0) && (fp8_mag != 0);

            // E4M3 NaN = 8'h7F (exp=15, man=7) or 8'hFF
            assign fp8_result[gi] = s2_fp8_nan[gi]  ? {fp8_rsign, 4'hF, 3'h7} :
                                    (fp8_mag == 0)   ? {(s2_fp8_eff_sub[gi] ? 1'b0 : s2_fp8_prod_sign[gi]), 7'b0} :
                                    fp8_is_ov        ? {fp8_rsign, 4'hE, 3'h7} : // max normal value (no Inf in E4M3)
                                    fp8_is_uf        ? {fp8_rsign, 7'b0} :
                                                       {fp8_rsign, fp8_fexp[3:0], fp8_fman};
        end
    endgenerate

    // =========================================================================
    // Output Mux
    // =========================================================================
    wire is_fp16_out = (s2_mode == 2'b01);
    wire is_fp8_out  = (s2_mode == 2'b10);

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0; valid_out <= 0;
            overflow <= 0; underflow <= 0; inexact <= 0;
        end else begin
            valid_out <= s2_valid;
            overflow <= 0; underflow <= 0; inexact <= 0;

            if (s2_valid) begin
                if (is_fp8_out) begin
                    // Pack 4 FP8 results
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
            end
        end
    end

endmodule
