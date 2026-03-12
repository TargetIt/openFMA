//============================================================================
// FMA Pipeline Stage 1: Multiplication
// - Sign computation
// - Exponent addition with bias correction
// - Mantissa multiplication via shared 48-bit multiplier
// - Supports FP32/FP16/FP8/INT8 through mode-based input muxing
//============================================================================
module fma_fp32_stage1 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [2:0]  mode,
    input  wire        valid_in,
    input  wire [31:0] A,
    input  wire [31:0] B,
    input  wire [31:0] C,

    // Pipeline outputs (registered)
    output reg         valid_s1,
    output reg  [2:0]  mode_s1,
    output reg         sign_product,    // Sign of A*B
    output reg         sign_c,          // Sign of C
    output reg  [9:0]  exp_sum,         // Biased exponent sum (10-bit for overflow)
    output reg  [8:0]  exp_c,           // Exponent of C
    output reg  [47:0] product,         // Mantissa product
    output reg  [23:0] mantissa_c,      // C mantissa with implicit bit
    output reg  [31:0] C_passthrough,   // C pass-through for INT8/FP8
    output reg  [31:0] A_passthrough,   // A pass-through for FP8 lanes
    output reg  [31:0] B_passthrough    // B pass-through for FP8 lanes
);

    // Combinational: extract fields and compute multiplication
    wire        sign_a, sign_b, sign_c_w;
    wire [8:0]  exp_a, exp_b, exp_c_w;  // 9-bit to handle bias subtraction
    wire [23:0] man_a, man_b, man_c;
    wire [9:0]  exp_sum_w;
    wire [47:0] product_w;

    // Shared multiplier inputs
    wire [23:0] mul_a_in, mul_b_in;

    //------------------------------------------------------------------------
    // Field Extraction & Multiplier Input Muxing
    //------------------------------------------------------------------------
    reg         r_sign_a, r_sign_b, r_sign_c;
    reg [8:0]   r_exp_a, r_exp_b, r_exp_c;
    reg [23:0]  r_man_a, r_man_b, r_man_c;
    reg [9:0]   r_exp_sum;

    always @(*) begin
        r_sign_a = 1'b0; r_sign_b = 1'b0; r_sign_c = 1'b0;
        r_exp_a  = 9'd0; r_exp_b  = 9'd0; r_exp_c  = 9'd0;
        r_man_a  = 24'd0; r_man_b  = 24'd0; r_man_c = 24'd0;
        r_exp_sum = 10'd0;

        case (mode)
        3'b000: begin // FP32
            r_sign_a = A[31];
            r_sign_b = B[31];
            r_sign_c = C[31];
            r_exp_a  = {1'b0, A[30:23]};
            r_exp_b  = {1'b0, B[30:23]};
            r_exp_c  = {1'b0, C[30:23]};
            r_man_a  = (A[30:23] == 8'b0) ? {1'b0, A[22:0]} : {1'b1, A[22:0]};
            r_man_b  = (B[30:23] == 8'b0) ? {1'b0, B[22:0]} : {1'b1, B[22:0]};
            r_man_c  = (C[30:23] == 8'b0) ? {1'b0, C[22:0]} : {1'b1, C[22:0]};
            r_exp_sum = {1'b0, r_exp_a} + {1'b0, r_exp_b} - 10'd127;
        end

        3'b001, 3'b100: begin // FP16 or FP16->FP32 Acc
            r_sign_a = A[15];
            r_sign_b = B[15];
            r_sign_c = C[15];
            r_exp_a  = {4'b0, A[14:10]};
            r_exp_b  = {4'b0, B[14:10]};
            r_exp_c  = {4'b0, C[14:10]};
            r_man_a  = (A[14:10] == 5'b0) ? {14'b0, A[9:0]}  : {14'b0, 1'b1, A[9:0]};
            r_man_b  = (B[14:10] == 5'b0) ? {14'b0, B[9:0]}  : {14'b0, 1'b1, B[9:0]};
            r_man_c  = (C[14:10] == 5'b0) ? {14'b0, C[9:0]}  : {14'b0, 1'b1, C[9:0]};
            // FP16 exp sum, then extend to FP32 range: +112
            r_exp_sum = {1'b0, r_exp_a} + {1'b0, r_exp_b} - 10'd15 + 10'd112;
            r_exp_c   = r_exp_c + 9'd112;
        end

        3'b010, 3'b101: begin // FP8x4 or FP8x4->FP32 Acc
            // Lane 0 only in stage1; other lanes handled in fp8_preprocessor
            r_sign_a = A[7];
            r_sign_b = B[7];
            r_sign_c = C[7];
            r_exp_a  = {5'b0, A[6:3]};
            r_exp_b  = {5'b0, B[6:3]};
            r_exp_c  = {5'b0, C[6:3]};
            r_man_a  = (A[6:3] == 4'b0) ? {21'b0, A[2:0]}  : {21'b0, 1'b1, A[2:0]};
            r_man_b  = (B[6:3] == 4'b0) ? {21'b0, B[2:0]}  : {21'b0, 1'b1, B[2:0]};
            r_man_c  = (C[6:3] == 4'b0) ? {21'b0, C[2:0]}  : {21'b0, 1'b1, C[2:0]};
            // FP8 exp sum, then extend to FP32 range: +113
            r_exp_sum = {1'b0, r_exp_a} + {1'b0, r_exp_b} - 10'd7 + 10'd113;
            r_exp_c   = r_exp_c + 9'd113;
        end

        3'b011: begin // INT8
            // Signed integer multiplication
            r_sign_a = A[7];
            r_sign_b = B[7];
            r_sign_c = C[15];
            // For INT8: use sign-extended values in multiplier
            r_man_a  = {{16{A[7]}}, A[7:0]};
            r_man_b  = {{16{B[7]}}, B[7:0]};
            r_man_c  = 24'd0;
            r_exp_a  = 9'd0;
            r_exp_b  = 9'd0;
            r_exp_c  = 9'd0;
            r_exp_sum = 10'd0;
        end

        default: begin
            r_sign_a = 1'b0; r_sign_b = 1'b0; r_sign_c = 1'b0;
            r_exp_a  = 9'd0; r_exp_b  = 9'd0; r_exp_c  = 9'd0;
            r_man_a  = 24'd0; r_man_b  = 24'd0; r_man_c  = 24'd0;
            r_exp_sum = 10'd0;
        end
        endcase
    end

    //------------------------------------------------------------------------
    // Shared 48-bit Multiplier Instance
    //------------------------------------------------------------------------
    multiplier_48bit u_mul (
        .mul_a   (r_man_a),
        .mul_b   (r_man_b),
        .product (product_w)
    );

    //------------------------------------------------------------------------
    // Pipeline Register
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1       <= 1'b0;
            mode_s1        <= 3'b0;
            sign_product   <= 1'b0;
            sign_c         <= 1'b0;
            exp_sum        <= 10'd0;
            exp_c          <= 9'd0;
            product        <= 48'd0;
            mantissa_c     <= 24'd0;
            C_passthrough  <= 32'd0;
            A_passthrough  <= 32'd0;
            B_passthrough  <= 32'd0;
        end else begin
            valid_s1       <= valid_in;
            mode_s1        <= mode;
            sign_product   <= r_sign_a ^ r_sign_b;
            sign_c         <= r_sign_c;
            exp_sum        <= r_exp_sum;
            exp_c          <= r_exp_c;
            product        <= product_w;
            mantissa_c     <= r_man_c;
            C_passthrough  <= C;
            A_passthrough  <= A;
            B_passthrough  <= B;
        end
    end

endmodule
