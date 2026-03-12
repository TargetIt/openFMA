//============================================================================
// Multi-Precision FMA Top Module
// Supports: FP32, FP16, FP8x4, INT8, FP16->FP32 Acc, FP8x4->FP32 Acc
//
// mode[2:0] encoding:
//   000 = FP32 FMA
//   001 = FP16 FMA
//   010 = FP8x4 FMA
//   011 = INT8 MAD
//   100 = FP16xFP16 -> FP32 Accumulate
//   101 = FP8x4 -> FP32 Accumulate
//
// 3-stage pipeline: Stage1=Multiply, Stage2=Align/CSA, Stage3=Normalize/Round
//============================================================================
module fma_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [2:0]  mode,        // Operation mode
    input  wire        valid_in,    // Input valid
    input  wire [31:0] A,           // Operand A
    input  wire [31:0] B,           // Operand B
    input  wire [31:0] C,           // Operand C / Addend
    input  wire        acc_en,      // Accumulate enable (Stage 3)
    input  wire [31:0] acc_in,      // External accumulator init value

    output wire [31:0] result,      // Computation result
    output wire [31:0] acc_out,     // Accumulator output (Stage 3)
    output wire        valid_out,   // Output valid
    output wire        overflow,    // Overflow flag
    output wire        underflow,   // Underflow flag
    output wire        inexact      // Inexact flag
);

    //========================================================================
    // Special Value Handler
    //========================================================================
    wire        is_special;
    wire [31:0] special_result;
    wire        special_overflow;

    special_value_handler u_special (
        .mode             (mode),
        .A                (A),
        .B                (B),
        .C                (C),
        .is_special       (is_special),
        .special_result   (special_result),
        .special_overflow (special_overflow)
    );

    //========================================================================
    // FP16 Preprocessor
    //========================================================================
    wire [31:0] fp16_fp32_a, fp16_fp32_b, fp16_fp32_c;

    fma_fp16_preprocessor u_fp16_pre (
        .fp16_a  (A[15:0]),
        .fp16_b  (B[15:0]),
        .fp16_c  (C[15:0]),
        .fp32_a  (fp16_fp32_a),
        .fp32_b  (fp16_fp32_b),
        .fp32_c  (fp16_fp32_c),
        .is_nan  (),
        .is_inf  (),
        .is_zero_a (),
        .is_zero_b ()
    );

    //========================================================================
    // FP8 Preprocessor
    //========================================================================
    wire [7:0]  fp8_prod_0, fp8_prod_1, fp8_prod_2, fp8_prod_3;
    wire [7:0]  fp8_exp_0, fp8_exp_1, fp8_exp_2, fp8_exp_3;
    wire [3:0]  fp8_signs_prod, fp8_signs_c;

    fma_fp8_preprocessor u_fp8_pre (
        .A          (A),
        .B          (B),
        .C          (C),
        .product_0  (fp8_prod_0),
        .product_1  (fp8_prod_1),
        .product_2  (fp8_prod_2),
        .product_3  (fp8_prod_3),
        .exp_sum_0  (fp8_exp_0),
        .exp_sum_1  (fp8_exp_1),
        .exp_sum_2  (fp8_exp_2),
        .exp_sum_3  (fp8_exp_3),
        .signs_product (fp8_signs_prod),
        .signs_c       (fp8_signs_c),
        .man_c_0    (),
        .man_c_1    (),
        .man_c_2    (),
        .man_c_3    (),
        .exp_c_0    (),
        .exp_c_1    (),
        .exp_c_2    (),
        .exp_c_3    ()
    );

    //========================================================================
    // INT8 Datapath
    //========================================================================
    wire [15:0] int8_result;
    wire        int8_overflow;

    fma_int8_datapath u_int8 (
        .A_int    (A[7:0]),
        .B_int    (B[7:0]),
        .C_int    (C[15:0]),
        .result   (int8_result),
        .overflow (int8_overflow)
    );

    //========================================================================
    // Input Mux: Route appropriate data to the FP32 pipeline
    //========================================================================
    reg  [31:0] pipe_A, pipe_B, pipe_C;

    always @(*) begin
        case (mode)
        3'b000: begin // FP32
            pipe_A = A;
            pipe_B = B;
            pipe_C = C;
        end
        3'b001, 3'b100: begin // FP16 or FP16->FP32 Acc
            pipe_A = A; // Stage1 handles FP16 field extraction directly
            pipe_B = B;
            pipe_C = C;
        end
        3'b010, 3'b101: begin // FP8x4 or FP8x4->FP32 Acc
            pipe_A = A; // Stage1 handles FP8 Lane0; others in preprocessor
            pipe_B = B;
            pipe_C = C;
        end
        3'b011: begin // INT8 - bypass pipeline, use INT8 datapath
            pipe_A = A;
            pipe_B = B;
            pipe_C = C;
        end
        default: begin
            pipe_A = A;
            pipe_B = B;
            pipe_C = C;
        end
        endcase
    end

    //========================================================================
    // Pipeline Stage 1: Multiplication
    //========================================================================
    wire        valid_s1;
    wire [2:0]  mode_s1;
    wire        sign_product_s1, sign_c_s1;
    wire [9:0]  exp_sum_s1;
    wire [8:0]  exp_c_s1;
    wire [47:0] product_s1;
    wire [23:0] mantissa_c_s1;
    wire [31:0] C_pass_s1, A_pass_s1, B_pass_s1;

    fma_fp32_stage1 u_stage1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .mode          (mode),
        .valid_in      (valid_in),
        .A             (pipe_A),
        .B             (pipe_B),
        .C             (pipe_C),
        .valid_s1      (valid_s1),
        .mode_s1       (mode_s1),
        .sign_product  (sign_product_s1),
        .sign_c        (sign_c_s1),
        .exp_sum       (exp_sum_s1),
        .exp_c         (exp_c_s1),
        .product       (product_s1),
        .mantissa_c    (mantissa_c_s1),
        .C_passthrough (C_pass_s1),
        .A_passthrough (A_pass_s1),
        .B_passthrough (B_pass_s1)
    );

    //========================================================================
    // Pipeline Stage 2: Alignment & CSA
    //========================================================================
    wire        valid_s2;
    wire [2:0]  mode_s2;
    wire        sign_result_s2;
    wire [9:0]  exp_tentative_s2;
    wire [49:0] csa_sum_s2, csa_carry_s2;
    wire        eff_sub_s2;
    wire        sticky_s2;
    wire [31:0] C_pass_s2, A_pass_s2, B_pass_s2;

    fma_fp32_stage2 u_stage2 (
        .clk              (clk),
        .rst_n            (rst_n),
        .valid_s1         (valid_s1),
        .mode_s1          (mode_s1),
        .sign_product     (sign_product_s1),
        .sign_c           (sign_c_s1),
        .exp_sum          (exp_sum_s1),
        .exp_c            (exp_c_s1),
        .product          (product_s1),
        .mantissa_c       (mantissa_c_s1),
        .C_passthrough    (C_pass_s1),
        .A_passthrough    (A_pass_s1),
        .B_passthrough    (B_pass_s1),
        .valid_s2         (valid_s2),
        .mode_s2          (mode_s2),
        .sign_result      (sign_result_s2),
        .exp_tentative    (exp_tentative_s2),
        .csa_sum          (csa_sum_s2),
        .csa_carry        (csa_carry_s2),
        .eff_sub          (eff_sub_s2),
        .sticky_s2        (sticky_s2),
        .C_passthrough_s2 (C_pass_s2),
        .A_passthrough_s2 (A_pass_s2),
        .B_passthrough_s2 (B_pass_s2)
    );

    //========================================================================
    // Pipeline Stage 3: Normalization & Rounding
    //========================================================================
    wire        valid_s3;
    wire [31:0] result_s3;
    wire        overflow_s3, underflow_s3, inexact_s3;

    fma_fp32_stage3 u_stage3 (
        .clk              (clk),
        .rst_n            (rst_n),
        .valid_s2         (valid_s2),
        .mode_s2          (mode_s2),
        .sign_result      (sign_result_s2),
        .exp_tentative    (exp_tentative_s2),
        .csa_sum          (csa_sum_s2),
        .csa_carry        (csa_carry_s2),
        .eff_sub          (eff_sub_s2),
        .sticky_s2        (sticky_s2),
        .C_passthrough_s2 (C_pass_s2),
        .valid_out        (valid_s3),
        .result           (result_s3),
        .overflow         (overflow_s3),
        .underflow        (underflow_s3),
        .inexact          (inexact_s3)
    );

    //========================================================================
    // Accumulator (Stage 3 modes)
    //========================================================================
    wire [31:0] acc_result;
    wire        acc_ovf;

    fma_accumulator u_acc (
        .clk          (clk),
        .rst_n        (rst_n),
        .acc_en       (acc_en),
        .acc_in       (acc_in),
        .fma_result   (result_s3),
        .valid_in     (valid_s3),
        .mode         (mode_s2),
        .acc_out      (acc_result),
        .acc_overflow (acc_ovf)
    );

    //========================================================================
    // Special Value Pipeline (delay to match 3-stage pipeline)
    //========================================================================
    reg        special_valid_d1, special_valid_d2, special_valid_d3;
    reg [31:0] special_result_d1, special_result_d2, special_result_d3;
    reg        special_ovf_d1, special_ovf_d2, special_ovf_d3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            special_valid_d1  <= 1'b0;
            special_valid_d2  <= 1'b0;
            special_valid_d3  <= 1'b0;
            special_result_d1 <= 32'b0;
            special_result_d2 <= 32'b0;
            special_result_d3 <= 32'b0;
            special_ovf_d1    <= 1'b0;
            special_ovf_d2    <= 1'b0;
            special_ovf_d3    <= 1'b0;
        end else begin
            special_valid_d1  <= is_special & valid_in;
            special_valid_d2  <= special_valid_d1;
            special_valid_d3  <= special_valid_d2;
            special_result_d1 <= special_result;
            special_result_d2 <= special_result_d1;
            special_result_d3 <= special_result_d2;
            special_ovf_d1    <= special_overflow;
            special_ovf_d2    <= special_ovf_d1;
            special_ovf_d3    <= special_ovf_d2;
        end
    end

    //========================================================================
    // INT8 Pipeline (delay to match 3-stage pipeline)
    //========================================================================
    reg        int8_valid_d1, int8_valid_d2, int8_valid_d3;
    reg [15:0] int8_result_d1, int8_result_d2, int8_result_d3;
    reg        int8_ovf_d1, int8_ovf_d2, int8_ovf_d3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            int8_valid_d1  <= 1'b0;
            int8_valid_d2  <= 1'b0;
            int8_valid_d3  <= 1'b0;
            int8_result_d1 <= 16'b0;
            int8_result_d2 <= 16'b0;
            int8_result_d3 <= 16'b0;
            int8_ovf_d1    <= 1'b0;
            int8_ovf_d2    <= 1'b0;
            int8_ovf_d3    <= 1'b0;
        end else begin
            int8_valid_d1  <= (mode == 3'b011) & valid_in;
            int8_valid_d2  <= int8_valid_d1;
            int8_valid_d3  <= int8_valid_d2;
            int8_result_d1 <= int8_result;
            int8_result_d2 <= int8_result_d1;
            int8_result_d3 <= int8_result_d2;
            int8_ovf_d1    <= int8_overflow;
            int8_ovf_d2    <= int8_ovf_d1;
            int8_ovf_d3    <= int8_ovf_d2;
        end
    end

    //========================================================================
    // Output Mux
    //========================================================================
    assign result    = special_valid_d3 ? special_result_d3 :
                       int8_valid_d3    ? {16'b0, int8_result_d3} :
                       result_s3;

    assign valid_out = special_valid_d3 | int8_valid_d3 | valid_s3;

    assign overflow  = special_valid_d3 ? special_ovf_d3 :
                       int8_valid_d3    ? int8_ovf_d3 :
                       overflow_s3 | acc_ovf;

    assign underflow = special_valid_d3 ? 1'b0 :
                       int8_valid_d3    ? 1'b0 :
                       underflow_s3;

    assign inexact   = special_valid_d3 ? 1'b0 :
                       int8_valid_d3    ? 1'b0 :
                       inexact_s3;

    assign acc_out   = acc_result;

endmodule
