//============================================================================
// FMA Pipeline Stage 2: Alignment and CSA Compression
// - Computes alignment shift amount
// - Aligns addend mantissa using shared 50-bit shifter
// - CSA compression of product and aligned addend
// - Handles effective subtraction (sign difference)
//============================================================================
module fma_fp32_stage2 (
    input  wire        clk,
    input  wire        rst_n,

    // From Stage 1
    input  wire        valid_s1,
    input  wire [2:0]  mode_s1,
    input  wire        sign_product,
    input  wire        sign_c,
    input  wire [9:0]  exp_sum,         // Exponent of product
    input  wire [8:0]  exp_c,           // Exponent of C
    input  wire [47:0] product,         // 48-bit mantissa product
    input  wire [23:0] mantissa_c,      // C mantissa with implicit bit
    input  wire [31:0] C_passthrough,
    input  wire [31:0] A_passthrough,
    input  wire [31:0] B_passthrough,

    // Pipeline outputs (registered)
    output reg         valid_s2,
    output reg  [2:0]  mode_s2,
    output reg         sign_result,     // Tentative sign
    output reg  [9:0]  exp_tentative,   // Tentative exponent
    output reg  [49:0] csa_sum,         // CSA sum output
    output reg  [49:0] csa_carry,       // CSA carry output
    output reg         eff_sub,         // Effective subtraction flag
    output reg         sticky_s2,       // Sticky bit from alignment
    output reg  [31:0] C_passthrough_s2,
    output reg  [31:0] A_passthrough_s2,
    output reg  [31:0] B_passthrough_s2
);

    //------------------------------------------------------------------------
    // Alignment Shift Amount Calculation
    //------------------------------------------------------------------------
    wire signed [10:0] shift_amt_signed;
    wire [7:0]         shift_amt;
    wire               product_larger; // Product exponent >= C exponent

    assign shift_amt_signed = {1'b0, exp_sum} - {2'b0, exp_c};
    assign product_larger   = ~shift_amt_signed[10]; // positive or zero
    assign shift_amt        = product_larger ?
                              (shift_amt_signed[7:0]) :
                              (-shift_amt_signed[7:0]);

    //------------------------------------------------------------------------
    // Prepare 50-bit operands for CSA
    //------------------------------------------------------------------------
    wire [49:0] product_extended;
    wire [49:0] addend_extended;
    wire [49:0] addend_shifted;
    wire        sticky_from_shift;

    // Extend product to 50 bits (place 48-bit product at MSBs)
    assign product_extended = {product, 2'b0};

    // Extend addend mantissa to 50 bits (binary point between bit 48 and 47,
    // matching product_extended format where product[47]->bit49, product[46]->bit48)
    assign addend_extended = {1'b0, mantissa_c, 25'b0};

    //------------------------------------------------------------------------
    // Shared 50-bit Shifter
    //------------------------------------------------------------------------
    wire [49:0] shifted_data;
    wire        sticky_bit;

    shifter_50bit u_shifter (
        .data_in   (product_larger ? addend_extended : product_extended),
        .shift_amt (shift_amt),
        .shift_dir (1'b0),  // Right shift
        .data_out  (shifted_data),
        .sticky_bit(sticky_bit)
    );

    //------------------------------------------------------------------------
    // Determine effective operation and prepare CSA inputs
    //------------------------------------------------------------------------
    wire effective_sub = sign_product ^ sign_c;

    wire [49:0] aligned_product, aligned_addend;

    assign aligned_product = product_larger ? product_extended : shifted_data;
    assign aligned_addend  = product_larger ? shifted_data     : addend_extended;

    // If effective subtraction, negate the addend for CSA
    wire [49:0] addend_for_csa;
    assign addend_for_csa = effective_sub ? (~aligned_addend) : aligned_addend;

    //------------------------------------------------------------------------
    // 3-2 CSA Instance
    //------------------------------------------------------------------------
    wire [49:0] csa_sum_w, csa_carry_w;

    csa_3_2 #(.WIDTH(50)) u_csa (
        .in_a  (aligned_product),
        .in_b  (addend_for_csa),
        .in_c  (effective_sub ? 50'b1 : 50'b0), // +1 for two's complement
        .sum   (csa_sum_w),
        .carry (csa_carry_w)
    );

    //------------------------------------------------------------------------
    // Tentative exponent: max of product exp and C exp
    //------------------------------------------------------------------------
    wire [9:0] exp_tent;
    assign exp_tent = product_larger ? exp_sum : {1'b0, exp_c};

    //------------------------------------------------------------------------
    // Pipeline Register
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2         <= 1'b0;
            mode_s2          <= 3'b0;
            sign_result      <= 1'b0;
            exp_tentative    <= 10'd0;
            csa_sum          <= 50'd0;
            csa_carry        <= 50'd0;
            eff_sub          <= 1'b0;
            sticky_s2        <= 1'b0;
            C_passthrough_s2 <= 32'd0;
            A_passthrough_s2 <= 32'd0;
            B_passthrough_s2 <= 32'd0;
        end else begin
            valid_s2         <= valid_s1;
            mode_s2          <= mode_s1;
            sign_result      <= product_larger ? sign_product : sign_c;
            exp_tentative    <= exp_tent;
            csa_sum          <= csa_sum_w;
            csa_carry        <= csa_carry_w;
            eff_sub          <= effective_sub;
            sticky_s2        <= sticky_bit;
            C_passthrough_s2 <= C_passthrough;
            A_passthrough_s2 <= A_passthrough;
            B_passthrough_s2 <= B_passthrough;
        end
    end

endmodule
