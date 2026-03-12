//============================================================================
// FP16 Preprocessor
// Extracts FP16 fields and maps to FP32 intermediate format
// Reuses FP32 datapath for computation
//============================================================================
module fma_fp16_preprocessor (
    input  wire [15:0] fp16_a,
    input  wire [15:0] fp16_b,
    input  wire [15:0] fp16_c,

    // FP32-format outputs for shared datapath
    output wire [31:0] fp32_a,      // FP16->FP32 extended A
    output wire [31:0] fp32_b,      // FP16->FP32 extended B
    output wire [31:0] fp32_c,      // FP16->FP32 extended C

    // Special value flags
    output wire        is_nan,
    output wire        is_inf,
    output wire        is_zero_a,
    output wire        is_zero_b
);

    // Field extraction
    wire        sign_a = fp16_a[15];
    wire [4:0]  exp_a  = fp16_a[14:10];
    wire [9:0]  man_a  = fp16_a[9:0];

    wire        sign_b = fp16_b[15];
    wire [4:0]  exp_b  = fp16_b[14:10];
    wire [9:0]  man_b  = fp16_b[9:0];

    wire        sign_c = fp16_c[15];
    wire [4:0]  exp_c  = fp16_c[14:10];
    wire [9:0]  man_c  = fp16_c[9:0];

    // Special value detection
    wire a_nan  = (exp_a == 5'h1F) && (man_a != 10'b0);
    wire b_nan  = (exp_b == 5'h1F) && (man_b != 10'b0);
    wire c_nan  = (exp_c == 5'h1F) && (man_c != 10'b0);
    wire a_inf  = (exp_a == 5'h1F) && (man_a == 10'b0);
    wire b_inf  = (exp_b == 5'h1F) && (man_b == 10'b0);

    assign is_nan    = a_nan | b_nan | c_nan;
    assign is_inf    = a_inf | b_inf;
    assign is_zero_a = (exp_a == 5'b0) && (man_a == 10'b0);
    assign is_zero_b = (exp_b == 5'b0) && (man_b == 10'b0);

    // FP16 -> FP32 conversion
    // Exponent: fp32_exp = fp16_exp + 112 (bias conversion: 127 - 15 = 112)
    // Mantissa: fp32_man = {fp16_man, 13'b0}
    wire [7:0] exp_a_fp32 = (exp_a == 5'b0) ? 8'b0 : ({3'b0, exp_a} + 8'd112);
    wire [7:0] exp_b_fp32 = (exp_b == 5'b0) ? 8'b0 : ({3'b0, exp_b} + 8'd112);
    wire [7:0] exp_c_fp32 = (exp_c == 5'b0) ? 8'b0 : ({3'b0, exp_c} + 8'd112);

    assign fp32_a = {sign_a, exp_a_fp32, man_a, 13'b0};
    assign fp32_b = {sign_b, exp_b_fp32, man_b, 13'b0};
    assign fp32_c = {sign_c, exp_c_fp32, man_c, 13'b0};

endmodule
