// =============================================================================
// Stage 0: FP32 FMA (Fused Multiply-Add) - Baseline Implementation
// Formula: result[31:0] = A[31:0] * B[31:0] + C[31:0]
// 3-stage pipeline: Stage1=Multiply, Stage2=Align/CSA, Stage3=Normalize/Round
// =============================================================================

module fma_fp32_stage0 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] A,
    input  wire [31:0] B,
    input  wire [31:0] C,
    output reg  [31:0] result,
    output reg         valid_out,
    output reg         overflow,
    output reg         underflow,
    output reg         inexact
);

    // =========================================================================
    // Input Field Extraction
    // =========================================================================
    wire        a_sign = A[31];
    wire [7:0]  a_exp  = A[30:23];
    wire [22:0] a_man  = A[22:0];

    wire        b_sign = B[31];
    wire [7:0]  b_exp  = B[30:23];
    wire [22:0] b_man  = B[22:0];

    wire        c_sign = C[31];
    wire [7:0]  c_exp  = C[30:23];
    wire [22:0] c_man  = C[22:0];

    // Special value detection
    wire a_is_zero = (a_exp == 8'h00) && (a_man == 23'h0);
    wire b_is_zero = (b_exp == 8'h00) && (b_man == 23'h0);
    wire c_is_zero = (c_exp == 8'h00) && (c_man == 23'h0);
    wire a_is_inf  = (a_exp == 8'hFF) && (a_man == 23'h0);
    wire b_is_inf  = (b_exp == 8'hFF) && (b_man == 23'h0);
    wire c_is_inf  = (c_exp == 8'hFF) && (c_man == 23'h0);
    wire a_is_nan  = (a_exp == 8'hFF) && (a_man != 23'h0);
    wire b_is_nan  = (b_exp == 8'hFF) && (b_man != 23'h0);
    wire c_is_nan  = (c_exp == 8'hFF) && (c_man != 23'h0);
    wire a_is_denorm = (a_exp == 8'h00) && (a_man != 23'h0);
    wire b_is_denorm = (b_exp == 8'h00) && (b_man != 23'h0);
    wire c_is_denorm = (c_exp == 8'h00) && (c_man != 23'h0);

    // Mantissa with implicit 1 (or 0 for denormals/zero)
    wire [23:0] a_mantissa = a_is_denorm ? {1'b0, a_man} : (a_is_zero ? 24'h0 : {1'b1, a_man});
    wire [23:0] b_mantissa = b_is_denorm ? {1'b0, b_man} : (b_is_zero ? 24'h0 : {1'b1, b_man});
    wire [23:0] c_mantissa = c_is_denorm ? {1'b0, c_man} : (c_is_zero ? 24'h0 : {1'b1, c_man});

    // Adjusted exponent for denormals (denormal exp treated as 1)
    wire [7:0] a_exp_adj = a_is_denorm ? 8'h01 : a_exp;
    wire [7:0] b_exp_adj = b_is_denorm ? 8'h01 : b_exp;
    wire [7:0] c_exp_adj = c_is_denorm ? 8'h01 : c_exp;

    // Special case result determination
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
    reg        s1_special;
    reg        s1_nan;
    reg        s1_inf;
    reg        s1_inf_sign;
    reg        s1_product_sign;
    reg        s1_c_sign;
    reg [8:0]  s1_exp_sum;       // 9-bit biased product exponent
    reg [7:0]  s1_c_exp;         // biased C exponent
    reg [47:0] s1_product;       // 24b x 24b = 48b (UQ2.46)
    reg [23:0] s1_c_mantissa;    // C mantissa (UQ1.23)
    reg        s1_prod_zero;     // product is zero

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid        <= 1'b0;
            s1_special      <= 1'b0;
            s1_nan          <= 1'b0;
            s1_inf          <= 1'b0;
            s1_inf_sign     <= 1'b0;
            s1_product_sign <= 1'b0;
            s1_c_sign       <= 1'b0;
            s1_exp_sum      <= 9'b0;
            s1_c_exp        <= 8'b0;
            s1_product      <= 48'b0;
            s1_c_mantissa   <= 24'b0;
            s1_prod_zero    <= 1'b0;
        end else begin
            s1_valid        <= valid_in;
            s1_special      <= special_case;
            s1_nan          <= nan_result;
            s1_inf          <= inf_result;
            s1_inf_sign     <= inf_sign;
            s1_product_sign <= product_sign;
            s1_c_sign       <= c_sign;
            // Exponent: A_exp + B_exp - 127 (bias correction)
            s1_exp_sum      <= {1'b0, a_exp_adj} + {1'b0, b_exp_adj} - 9'd127;
            s1_c_exp        <= c_exp_adj;
            // 48-bit multiplier: {1,A[22:0]} x {1,B[22:0]} -> UQ2.46
            s1_product      <= a_mantissa * b_mantissa;
            s1_c_mantissa   <= c_mantissa;
            s1_prod_zero    <= a_is_zero | b_is_zero;
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Align / CSA
    // =========================================================================
    // Product is 48 bits in UQ2.46 format.
    // Extended to 50 bits (UQ2.48): product_aligned = {product, 2'b0}
    // C mantissa is 24 bits in UQ1.23, placed as UQ1.48 in 50 bits:
    //   c_base = {1'b0, c_mantissa, 25'b0}
    // Both have binary point between bits 48 and 47.
    // Alignment shift based on exp_diff = exp_sum - c_exp.

    wire signed [9:0] exp_diff = $signed({1'b0, s1_exp_sum}) - $signed({2'b0, s1_c_exp});

    reg [49:0] product_aligned;
    reg [49:0] c_aligned;
    reg [8:0]  align_exp;

    always @(*) begin
        if (exp_diff >= 0) begin
            // Product exponent >= C exponent: shift C right
            product_aligned = {s1_product, 2'b0};
            if (exp_diff > 49)
                c_aligned = 50'b0;
            else
                c_aligned = ({1'b0, s1_c_mantissa, 25'b0}) >> exp_diff;
            align_exp = s1_exp_sum;
        end else begin
            // C exponent > product exponent: shift product right
            if ((-exp_diff) > 49)
                product_aligned = 50'b0;
            else
                product_aligned = ({s1_product, 2'b0}) >> (-exp_diff);
            c_aligned = {1'b0, s1_c_mantissa, 25'b0};
            align_exp = {1'b0, s1_c_exp};
        end
    end

    // Effective operation: subtract if signs differ
    wire eff_sub = s1_product_sign ^ s1_c_sign;

    // CSA: 3-2 Carry Save Adder (50-bit)
    // For subtraction, use two's complement of c_aligned
    wire [49:0] csa_op_a = product_aligned;
    wire [49:0] csa_op_b = eff_sub ? (~c_aligned) : c_aligned;
    wire        csa_cin  = eff_sub ? 1'b1 : 1'b0;

    // 3-input CSA: sum and carry such that sum + carry = op_a + op_b + cin
    wire [49:0] cin_ext = {49'b0, csa_cin};
    wire [49:0] csa_sum   = csa_op_a ^ csa_op_b ^ cin_ext;
    wire [49:0] csa_carry = ((csa_op_a & csa_op_b) | (csa_op_a & cin_ext) | (csa_op_b & cin_ext)) << 1;

    // Stage 2 pipeline registers
    reg        s2_valid;
    reg        s2_special;
    reg        s2_nan;
    reg        s2_inf;
    reg        s2_inf_sign;
    reg        s2_product_sign;
    reg        s2_c_sign;
    reg        s2_eff_sub;
    reg [8:0]  s2_exp;
    reg [49:0] s2_sum;
    reg [49:0] s2_carry;
    reg        s2_prod_zero;
    reg        s2_c_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid        <= 1'b0;
            s2_special      <= 1'b0;
            s2_nan          <= 1'b0;
            s2_inf          <= 1'b0;
            s2_inf_sign     <= 1'b0;
            s2_product_sign <= 1'b0;
            s2_c_sign       <= 1'b0;
            s2_eff_sub      <= 1'b0;
            s2_exp          <= 9'b0;
            s2_sum          <= 50'b0;
            s2_carry        <= 50'b0;
            s2_prod_zero    <= 1'b0;
            s2_c_zero       <= 1'b0;
        end else begin
            s2_valid        <= s1_valid;
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
            s2_c_zero       <= (s1_c_mantissa == 24'h0);
        end
    end

    // =========================================================================
    // Pipeline Stage 3: Normalize / Round
    // =========================================================================

    // Final addition: 50-bit sum + 50-bit carry -> 51-bit result
    wire [50:0] raw_result = {1'b0, s2_sum} + {1'b0, s2_carry};

    // For subtraction: bit 50 = carry out
    //   carry=1 means product >= C (positive), magnitude = raw_result[49:0]
    //   carry=0 means product < C (negative), magnitude = ~raw_result[49:0] + 1
    // For addition: raw_result[50:0] is the positive magnitude (may use all 51 bits)
    wire sub_borrow = s2_eff_sub & ~raw_result[50];

    wire [50:0] magnitude = s2_eff_sub ?
        (raw_result[50] ? {1'b0, raw_result[49:0]} : ({1'b0, ~raw_result[49:0]} + 51'd1)) :
        raw_result;

    wire result_sign = sub_borrow ? s2_c_sign : s2_product_sign;

    // Leading zero detection (LZD) on magnitude[50:0]
    reg [5:0] lzd_count;
    integer k;
    always @(*) begin
        lzd_count = 6'd51;  // default: all zeros
        for (k = 0; k <= 50; k = k + 1) begin
            if (magnitude[k])
                lzd_count = 50 - k;
        end
    end

    // Normalization: shift left by lzd_count so leading 1 is at bit 50
    // norm_mantissa[50] = 1, [49:27] = 23-bit fraction, [26]=guard, [25]=round, [24:0]=sticky
    wire [50:0] norm_mantissa = magnitude << lzd_count;

    // Biased result exponent:
    // Leading 1 at bit P (from LSB), biased_exp = align_exp + P - 48
    // Since P = 50 - lzd_count: biased_exp = align_exp + 2 - lzd_count
    wire signed [9:0] norm_exp_signed = $signed({1'b0, s2_exp}) + $signed(10'sd2) - $signed({4'b0, lzd_count});
    wire [8:0] norm_exp = norm_exp_signed[8:0];

    // Rounding: Round to Nearest Even (RNE)
    wire [22:0] trunc_mantissa = norm_mantissa[49:27];
    wire        guard_bit  = norm_mantissa[26];
    wire        round_bit  = norm_mantissa[25];
    wire        sticky_bit = |norm_mantissa[24:0];

    wire round_up = guard_bit & (round_bit | sticky_bit | trunc_mantissa[0]);

    wire [23:0] rounded_mantissa = {1'b0, trunc_mantissa} + {23'b0, round_up};

    // Adjust if rounding causes mantissa overflow (1.111...1 + 1 = 10.000...0)
    wire [22:0] final_mantissa = rounded_mantissa[23] ? rounded_mantissa[23:1] : rounded_mantissa[22:0];
    wire [8:0]  final_exp = rounded_mantissa[23] ? norm_exp + 9'd1 : norm_exp;

    wire final_inexact = guard_bit | round_bit | sticky_bit;

    // Output assembly
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
                    // NaN
                    result <= 32'h7FC00000;
                end else if (s2_inf) begin
                    // Infinity
                    result <= {s2_inf_sign, 8'hFF, 23'h0};
                end else if (magnitude == 51'b0) begin
                    // Zero result
                    result <= {(s2_eff_sub ? 1'b0 : s2_product_sign), 31'b0};
                end else if (final_exp >= 9'd255) begin
                    // Overflow -> Infinity
                    result   <= {result_sign, 8'hFF, 23'h0};
                    overflow <= 1'b1;
                    inexact  <= 1'b1;
                end else if (norm_exp_signed <= 0) begin
                    // Underflow -> Zero (simplified: no denormal output)
                    result    <= {result_sign, 31'b0};
                    underflow <= 1'b1;
                    inexact   <= final_inexact;
                end else begin
                    // Normal result
                    result  <= {result_sign, final_exp[7:0], final_mantissa};
                    inexact <= final_inexact;
                end
            end
        end
    end

endmodule
