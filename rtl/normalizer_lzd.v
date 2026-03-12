//============================================================================
// Normalizer with Leading Zero Detector (LZD)
// Detects leading zeros and normalizes the mantissa
//============================================================================
module normalizer_lzd (
    input  wire [50:0] data_in,         // Input from adder (51-bit)
    output wire [49:0] normalized,      // Normalized mantissa
    output wire [7:0]  shift_count,     // Number of positions shifted
    output wire        is_zero          // All-zero flag
);

    reg [7:0]  lzd_count;
    reg [49:0] norm_out;
    reg        zero_flag;

    integer i;

    always @(*) begin
        lzd_count = 8'd0;
        norm_out  = data_in[49:0];
        zero_flag = 1'b0;

        if (data_in[50]) begin
            // Overflow bit set, shift right by 1
            lzd_count = 8'd255; // Special: indicates right-shift needed
            norm_out  = data_in[50:1];
        end else if (data_in[49:0] == 50'b0) begin
            zero_flag = 1'b1;
            lzd_count = 8'd50;
        end else begin
            // Count leading zeros: iterate from LSB to MSB.
            // Last matching assignment wins, giving position of the MSB.
            lzd_count = 8'd50;
            for (i = 0; i <= 49; i = i + 1) begin
                if (data_in[i]) begin
                    lzd_count = 8'd49 - i[7:0];
                end
            end
            // Normalize: left shift to remove leading zeros
            norm_out = data_in[49:0] << lzd_count;
        end
    end

    assign normalized  = norm_out;
    assign shift_count = lzd_count;
    assign is_zero     = zero_flag;

endmodule
