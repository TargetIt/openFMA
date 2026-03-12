// =============================================================================
// Stage 1: FP32 + FP16 FMA (Fused Multiply-Add)
// Supports: mode 00=FP32, mode 01=FP16
// Incremental on Stage 0: reuses multiplier, shifter, CSA, adder
// 3-stage pipeline: Stage1=Multiply, Stage2=Align/CSA, Stage3=Normalize/Round
// =============================================================================

module fma_fp16_stage1 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [1:0]  mode,       // 00=FP32, 01=FP16
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

    // =========================================================================
    // Unified Input Field Extraction
    // =========================================================================
    // FP32 fields
    wire        a32_sign = A[31];
    wire [7:0]  a32_exp  = A[30:23];
    wire [22:0] a32_man  = A[22:0];
    wire        b32_sign = B[31];
    wire [7:0]  b32_exp  = B[30:23];
    wire [22:0] b32_man  = B[22:0];
    wire        c32_sign = C[31];
    wire [7:0]  c32_exp  = C[30:23];
    wire [22:0] c32_man  = C[22:0];

    // FP16 fields
    wire        a16_sign = A[15];
    wire [4:0]  a16_exp  = A[14:10];
    wire [9:0]  a16_man  = A[9:0];
    wire        b16_sign = B[15];
    wire [4:0]  b16_exp  = B[14:10];
    wire [9:0]  b16_man  = B[9:0];
    wire        c16_sign = C[15];
    wire [4:0]  c16_exp  = C[14:10];
    wire [9:0]  c16_man  = C[9:0];

    // Unified fields (selected by mode)
    wire        a_sign = is_fp16 ? a16_sign : a32_sign;
    wire        b_sign = is_fp16 ? b16_sign : b32_sign;
    wire        c_sign = is_fp16 ? c16_sign : c32_sign;

    // Special value detection - FP32
    wire a32_is_zero   = (a32_exp == 8'h00) && (a32_man == 23'h0);
    wire b32_is_zero   = (b32_exp == 8'h00) && (b32_man == 23'h0);
    wire c32_is_zero   = (c32_exp == 8'h00) && (c32_man == 23'h0);
    wire a32_is_inf    = (a32_exp == 8'hFF) && (a32_man == 23'h0);
    wire b32_is_inf    = (b32_exp == 8'hFF) && (b32_man == 23'h0);
    wire c32_is_inf    = (c32_exp == 8'hFF) && (c32_man == 23'h0);
    wire a32_is_nan    = (a32_exp == 8'hFF) && (a32_man != 23'h0);
    wire b32_is_nan    = (b32_exp == 8'hFF) && (b32_man != 23'h0);
    wire c32_is_nan    = (c32_exp == 8'hFF) && (c32_man != 23'h0);
    wire a32_is_denorm = (a32_exp == 8'h00) && (a32_man != 23'h0);
    wire b32_is_denorm = (b32_exp == 8'h00) && (b32_man != 23'h0);
    wire c32_is_denorm = (c32_exp == 8'h00) && (c32_man != 23'h0);

    // Special value detection - FP16
    wire a16_is_zero   = (a16_exp == 5'h00) && (a16_man == 10'h0);
    wire b16_is_zero   = (b16_exp == 5'h00) && (b16_man == 10'h0);
    wire c16_is_zero   = (c16_exp == 5'h00) && (c16_man == 10'h0);
    wire a16_is_inf    = (a16_exp == 5'h1F) && (a16_man == 10'h0);
    wire b16_is_inf    = (b16_exp == 5'h1F) && (b16_man == 10'h0);
    wire c16_is_inf    = (c16_exp == 5'h1F) && (c16_man == 10'h0);
    wire a16_is_nan    = (a16_exp == 5'h1F) && (a16_man != 10'h0);
    wire b16_is_nan    = (b16_exp == 5'h1F) && (b16_man != 10'h0);
    wire c16_is_nan    = (c16_exp == 5'h1F) && (c16_man != 10'h0);
    wire a16_is_denorm = (a16_exp == 5'h00) && (a16_man != 10'h0);
    wire b16_is_denorm = (b16_exp == 5'h00) && (b16_man != 10'h0);
    wire c16_is_denorm = (c16_exp == 5'h00) && (c16_man != 10'h0);

    // Unified special flags
    wire a_is_zero   = is_fp16 ? a16_is_zero   : a32_is_zero;
    wire b_is_zero   = is_fp16 ? b16_is_zero   : b32_is_zero;
    wire c_is_zero   = is_fp16 ? c16_is_zero   : c32_is_zero;
    wire a_is_inf    = is_fp16 ? a16_is_inf    : a32_is_inf;
    wire b_is_inf    = is_fp16 ? b16_is_inf    : b32_is_inf;
    wire c_is_inf    = is_fp16 ? c16_is_inf    : c32_is_inf;
    wire a_is_nan    = is_fp16 ? a16_is_nan    : a32_is_nan;
    wire b_is_nan    = is_fp16 ? b16_is_nan    : b32_is_nan;
    wire c_is_nan    = is_fp16 ? c16_is_nan    : c32_is_nan;
    wire a_is_denorm = is_fp16 ? a16_is_denorm : a32_is_denorm;
    wire b_is_denorm = is_fp16 ? b16_is_denorm : b32_is_denorm;
    wire c_is_denorm = is_fp16 ? c16_is_denorm : c32_is_denorm;

    // =========================================================================
    // Unified Mantissa and Exponent (converted to FP32 internal format)
    // =========================================================================
    // FP32 mantissa: 24-bit (1 + 23)
    wire [23:0] a32_mantissa = a32_is_denorm ? {1'b0, a32_man} :
                               (a32_is_zero ? 24'h0 : {1'b1, a32_man});
    wire [23:0] b32_mantissa = b32_is_denorm ? {1'b0, b32_man} :
                               (b32_is_zero ? 24'h0 : {1'b1, b32_man});
    wire [23:0] c32_mantissa = c32_is_denorm ? {1'b0, c32_man} :
                               (c32_is_zero ? 24'h0 : {1'b1, c32_man});

    // FP16 mantissa: 11-bit (1 + 10), extended to 24-bit for FP32 path
    wire [10:0] a16_mantissa_raw = a16_is_denorm ? {1'b0, a16_man} :
                                   (a16_is_zero ? 11'h0 : {1'b1, a16_man});
    wire [10:0] b16_mantissa_raw = b16_is_denorm ? {1'b0, b16_man} :
                                   (b16_is_zero ? 11'h0 : {1'b1, b16_man});
    wire [10:0] c16_mantissa_raw = c16_is_denorm ? {1'b0, c16_man} :
                                   (c16_is_zero ? 11'h0 : {1'b1, c16_man});

    // FP16 mantissa mapped to 24-bit (zero-padded at LSB)
    wire [23:0] a16_mantissa = {a16_mantissa_raw, 13'b0};
    wire [23:0] b16_mantissa = {b16_mantissa_raw, 13'b0};
    wire [23:0] c16_mantissa = {c16_mantissa_raw, 13'b0};

    // Unified mantissa
    wire [23:0] a_mantissa = is_fp16 ? a16_mantissa : a32_mantissa;
    wire [23:0] b_mantissa = is_fp16 ? b16_mantissa : b32_mantissa;
    wire [23:0] c_mantissa = is_fp16 ? c16_mantissa : c32_mantissa;

    // FP16 exponent -> FP32 biased exponent
    // FP16: bias=15, FP32: bias=127, offset = 127 - 15 = 112
    wire [7:0] a16_exp_fp32 = a16_is_denorm ? 8'd113 : {3'b0, a16_exp} + 8'd112;
    wire [7:0] b16_exp_fp32 = b16_is_denorm ? 8'd113 : {3'b0, b16_exp} + 8'd112;
    wire [7:0] c16_exp_fp32 = c16_is_denorm ? 8'd113 : {3'b0, c16_exp} + 8'd112;

    // FP32 adjusted exponent
    wire [7:0] a32_exp_adj = a32_is_denorm ? 8'h01 : a32_exp;
    wire [7:0] b32_exp_adj = b32_is_denorm ? 8'h01 : b32_exp;
    wire [7:0] c32_exp_adj = c32_is_denorm ? 8'h01 : c32_exp;

    // Unified exponent (FP32 biased)
    wire [7:0] a_exp = is_fp16 ? a16_exp_fp32 : a32_exp_adj;
    wire [7:0] b_exp = is_fp16 ? b16_exp_fp32 : b32_exp_adj;
    wire [7:0] c_exp = is_fp16 ? c16_exp_fp32 : c32_exp_adj;

    // Special case flags
    wire product_sign = a_sign ^ b_sign;
    wire special_case = a_is_nan | b_is_nan | c_is_nan |
                        a_is_inf | b_is_inf | c_is_inf;
    wire nan_result = a_is_nan | b_is_nan | c_is_nan |
                      (a_is_inf & b_is_zero) | (b_is_inf & a_is_zero) |
                      ((a_is_inf | b_is_inf) & c_is_inf & (product_sign ^ c_sign));
    wire inf_result = ~nan_result & (a_is_inf | b_is_inf | c_is_inf);
    wire inf_sign   = (a_is_inf | b_is_inf) ? product_sign : c_sign;

    // =========================================================================
    // Pipeline Stage 1: Multiply
    // =========================================================================
    reg        s1_valid;
    reg [1:0]  s1_mode;
    reg        s1_special, s1_nan, s1_inf, s1_inf_sign;
    reg        s1_product_sign, s1_c_sign;
    reg [8:0]  s1_exp_sum;
    reg [7:0]  s1_c_exp;
    reg [47:0] s1_product;
    reg [23:0] s1_c_mantissa;
    reg        s1_prod_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 0; s1_mode <= 0;
            s1_special <= 0; s1_nan <= 0; s1_inf <= 0; s1_inf_sign <= 0;
            s1_product_sign <= 0; s1_c_sign <= 0;
            s1_exp_sum <= 0; s1_c_exp <= 0;
            s1_product <= 0; s1_c_mantissa <= 0;
            s1_prod_zero <= 0;
        end else begin
            s1_valid        <= valid_in;
            s1_mode         <= mode;
            s1_special      <= special_case;
            s1_nan          <= nan_result;
            s1_inf          <= inf_result;
            s1_inf_sign     <= inf_sign;
            s1_product_sign <= product_sign;
            s1_c_sign       <= c_sign;
            s1_exp_sum      <= {1'b0, a_exp} + {1'b0, b_exp} - 9'd127;
            s1_c_exp        <= c_exp;
            // 48-bit multiplier (shared): FP16 uses lower bits, FP32 uses full
            s1_product      <= a_mantissa * b_mantissa;
            s1_c_mantissa   <= c_mantissa;
            s1_prod_zero    <= a_is_zero | b_is_zero;
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Align / CSA
    // =========================================================================
    wire signed [9:0] exp_diff = $signed({1'b0, s1_exp_sum}) - $signed({2'b0, s1_c_exp});

    reg [49:0] product_aligned;
    reg [49:0] c_aligned;
    reg [8:0]  align_exp;

    always @(*) begin
        if (exp_diff >= 0) begin
            product_aligned = {s1_product, 2'b0};
            if (exp_diff > 49)
                c_aligned = 50'b0;
            else
                c_aligned = ({1'b0, s1_c_mantissa, 25'b0}) >> exp_diff;
            align_exp = s1_exp_sum;
        end else begin
            if ((-exp_diff) > 49)
                product_aligned = 50'b0;
            else
                product_aligned = ({s1_product, 2'b0}) >> (-exp_diff);
            c_aligned = {1'b0, s1_c_mantissa, 25'b0};
            align_exp = {1'b0, s1_c_exp};
        end
    end

    wire eff_sub = s1_product_sign ^ s1_c_sign;

    wire [49:0] csa_op_a = product_aligned;
    wire [49:0] csa_op_b = eff_sub ? (~c_aligned) : c_aligned;
    wire        csa_cin  = eff_sub;
    wire [49:0] cin_ext  = {49'b0, csa_cin};
    wire [49:0] csa_sum   = csa_op_a ^ csa_op_b ^ cin_ext;
    wire [49:0] csa_carry = ((csa_op_a & csa_op_b) | (csa_op_a & cin_ext) | (csa_op_b & cin_ext)) << 1;

    reg        s2_valid;
    reg [1:0]  s2_mode;
    reg        s2_special, s2_nan, s2_inf, s2_inf_sign;
    reg        s2_product_sign, s2_c_sign, s2_eff_sub;
    reg [8:0]  s2_exp;
    reg [49:0] s2_sum, s2_carry;
    reg        s2_prod_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 0; s2_mode <= 0;
            s2_special <= 0; s2_nan <= 0; s2_inf <= 0; s2_inf_sign <= 0;
            s2_product_sign <= 0; s2_c_sign <= 0; s2_eff_sub <= 0;
            s2_exp <= 0; s2_sum <= 0; s2_carry <= 0;
            s2_prod_zero <= 0;
        end else begin
            s2_valid        <= s1_valid;
            s2_mode         <= s1_mode;
            s2_special      <= s1_special;
            s2_nan          <= s1_nan;
            s2_inf          <= s1_inf;
            s2_inf_sign     <= s1_inf_sign;
            s2_product_sign <= s1_product_sign;
            s2_c_sign       <= s1_c_sign;
            s2_eff_sub      <= eff_sub;
            s2_exp          <= align_exp;
            s2_sum          <= csa_sum;
            s2_carry        <= csa_carry;
            s2_prod_zero    <= s1_prod_zero;
        end
    end

    // =========================================================================
    // Pipeline Stage 3: Normalize / Round
    // =========================================================================
    wire [50:0] raw_result = {1'b0, s2_sum} + {1'b0, s2_carry};
    wire sub_borrow = s2_eff_sub & ~raw_result[50];

    wire [50:0] magnitude = s2_eff_sub ?
        (raw_result[50] ? {1'b0, raw_result[49:0]} : ({1'b0, ~raw_result[49:0]} + 51'd1)) :
        raw_result;

    wire result_sign = sub_borrow ? s2_c_sign : s2_product_sign;

    // LZD
    reg [5:0] lzd_count;
    integer k;
    always @(*) begin
        lzd_count = 6'd51;
        for (k = 0; k <= 50; k = k + 1) begin
            if (magnitude[k])
                lzd_count = 50 - k;
        end
    end

    wire [50:0] norm_mantissa = magnitude << lzd_count;
    wire signed [9:0] norm_exp_signed = $signed({1'b0, s2_exp}) + 10'sd2 - $signed({4'b0, lzd_count});
    wire [8:0] norm_exp = norm_exp_signed[8:0];

    // Rounding (RNE)
    wire [22:0] trunc_mantissa = norm_mantissa[49:27];
    wire        guard_bit  = norm_mantissa[26];
    wire        round_bit  = norm_mantissa[25];
    wire        sticky_bit = |norm_mantissa[24:0];
    wire round_up = guard_bit & (round_bit | sticky_bit | trunc_mantissa[0]);

    wire [23:0] rounded_mantissa = {1'b0, trunc_mantissa} + {23'b0, round_up};
    wire [22:0] final_mantissa = rounded_mantissa[23] ? rounded_mantissa[23:1] : rounded_mantissa[22:0];
    wire [8:0]  final_exp = rounded_mantissa[23] ? norm_exp + 9'd1 : norm_exp;
    wire final_inexact = guard_bit | round_bit | sticky_bit;

    // FP32 result assembly
    wire [31:0] fp32_result;
    wire        fp32_overflow;
    wire        fp32_underflow;

    assign fp32_overflow  = (final_exp >= 9'd255) && (magnitude != 51'b0);
    assign fp32_underflow = (norm_exp_signed <= 0) && (magnitude != 51'b0);
    assign fp32_result = (magnitude == 51'b0)  ? {(s2_eff_sub ? 1'b0 : s2_product_sign), 31'b0} :
                         fp32_overflow         ? {result_sign, 8'hFF, 23'h0} :
                         fp32_underflow        ? {result_sign, 31'b0} :
                                                 {result_sign, final_exp[7:0], final_mantissa};

    // FP16 result: convert FP32 biased exponent back to FP16
    // FP32 biased exp -> FP16 biased exp: subtract 112
    wire [8:0] fp16_exp_biased = final_exp - 9'd112;
    wire       fp16_overflow  = (fp16_exp_biased >= 9'd31) && (magnitude != 51'b0);
    wire       fp16_underflow = (fp16_exp_biased[8] || fp16_exp_biased == 9'd0) && (magnitude != 51'b0);

    // FP16 mantissa: take top 10 bits of the 23-bit mantissa
    wire [9:0] fp16_mantissa = final_mantissa[22:13];

    wire [15:0] fp16_result = (magnitude == 51'b0)  ? {(s2_eff_sub ? 1'b0 : s2_product_sign), 15'b0} :
                              fp16_overflow          ? {result_sign, 5'h1F, 10'h0} :
                              fp16_underflow         ? {result_sign, 15'b0} :
                                                       {result_sign, fp16_exp_biased[4:0], fp16_mantissa};

    // =========================================================================
    // Output Mux
    // =========================================================================
    wire is_fp16_mode = (s2_mode == 2'b01);

    // NaN/Inf constants
    wire [31:0] fp32_nan = 32'h7FC00000;
    wire [15:0] fp16_nan = 16'h7E00;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result    <= 32'b0;
            valid_out <= 1'b0;
            overflow  <= 1'b0;
            underflow <= 1'b0;
            inexact   <= 1'b0;
        end else begin
            valid_out <= s2_valid;
            overflow  <= 1'b0;
            underflow <= 1'b0;
            inexact   <= 1'b0;

            if (s2_valid) begin
                if (s2_nan) begin
                    result <= is_fp16_mode ? {16'b0, fp16_nan} : fp32_nan;
                end else if (s2_inf) begin
                    if (is_fp16_mode)
                        result <= {16'b0, s2_inf_sign, 5'h1F, 10'h0};
                    else
                        result <= {s2_inf_sign, 8'hFF, 23'h0};
                end else begin
                    if (is_fp16_mode) begin
                        result    <= {16'b0, fp16_result};
                        overflow  <= fp16_overflow;
                        underflow <= fp16_underflow;
                    end else begin
                        result    <= fp32_result;
                        overflow  <= fp32_overflow;
                        underflow <= fp32_underflow;
                    end
                    inexact <= final_inexact;
                end
            end
        end
    end

endmodule
