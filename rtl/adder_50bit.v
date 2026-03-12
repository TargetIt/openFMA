//============================================================================
// 50-bit Adder
// Used for final addition after CSA compression
//============================================================================
module adder_50bit (
    input  wire [49:0] in_a,
    input  wire [49:0] in_b,
    input  wire        cin,
    output wire [50:0] result   // 51-bit to capture carry-out
);

    assign result = {1'b0, in_a} + {1'b0, in_b} + {50'b0, cin};

endmodule
