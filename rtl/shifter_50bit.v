//============================================================================
// 50-bit Right Shifter
// Used for alignment shifting of addend mantissa
//============================================================================
module shifter_50bit (
    input  wire [49:0] data_in,     // Input data
    input  wire [7:0]  shift_amt,   // Shift amount (unsigned)
    input  wire        shift_dir,   // 0 = right shift, 1 = left shift
    output wire [49:0] data_out,    // Shifted output
    output wire        sticky_bit   // OR of all shifted-out bits (for rounding)
);

    reg [49:0] shifted;
    reg        sticky;

    always @(*) begin
        sticky = 1'b0;
        if (shift_dir == 1'b0) begin
            // Right shift
            if (shift_amt >= 8'd50) begin
                sticky  = |data_in;
                shifted = 50'b0;
            end else begin
                sticky  = |(data_in & ((50'b1 << shift_amt) - 50'b1));
                shifted = data_in >> shift_amt;
            end
        end else begin
            // Left shift
            if (shift_amt >= 8'd50) begin
                shifted = 50'b0;
            end else begin
                shifted = data_in << shift_amt;
            end
        end
    end

    assign data_out   = shifted;
    assign sticky_bit = sticky;

endmodule
