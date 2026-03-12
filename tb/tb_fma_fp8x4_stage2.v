// =============================================================================
// Testbench for Stage 2: FP32 + FP16 + FP8x4 FMA
// =============================================================================
`timescale 1ns / 1ps

module tb_fma_fp8x4_stage2;

    reg         clk;
    reg         rst_n;
    reg         valid_in;
    reg  [1:0]  mode;
    reg  [31:0] A, B, C;
    wire [31:0] result;
    wire        valid_out, overflow, underflow, inexact;

    fma_fp8x4_stage2 dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .mode(mode), .A(A), .B(B), .C(C),
        .result(result), .valid_out(valid_out),
        .overflow(overflow), .underflow(underflow), .inexact(inexact)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // FP32 constants
    localparam FP32_ONE   = 32'h3F800000;
    localparam FP32_TWO   = 32'h40000000;
    localparam FP32_THREE = 32'h40400000;
    localparam FP32_FOUR  = 32'h40800000;
    localparam FP32_TEN   = 32'h41200000;
    localparam FP32_ZERO  = 32'h00000000;

    // FP16 constants
    localparam FP16_ONE   = 32'h00003C00;
    localparam FP16_TWO   = 32'h00004000;
    localparam FP16_THREE = 32'h00004200;
    localparam FP16_FOUR  = 32'h00004400;
    localparam FP16_TEN   = 32'h00004900;
    localparam FP16_ZERO  = 32'h00000000;

    // FP8 E4M3 encoding helpers
    // E4M3: [7]=sign, [6:3]=exp(bias=7), [2:0]=man
    // 1.0 = exp=7(0111), man=0(000) -> 8'h38
    // 2.0 = exp=8(1000), man=0(000) -> 8'h40
    // 3.0 = exp=8(1000), man=4(100) -> 8'h44
    // 0.0 = 8'h00
    // -1.0 = 8'hB8

    // Pack 4 FP8 values into 32 bits: {lane3, lane2, lane1, lane0}
    function [31:0] pack_fp8;
        input [7:0] v0, v1, v2, v3;
        pack_fp8 = {v3, v2, v1, v0};
    endfunction

    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    task apply_input;
        input [1:0] m;
        input [31:0] a_val, b_val, c_val;
        begin
            @(posedge clk);
            mode <= m; A <= a_val; B <= b_val; C <= c_val;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task wait_result;
        begin
            @(posedge clk); @(posedge clk); @(posedge clk);
        end
    endtask

    task check_result;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result == expected) begin
                $display("PASS: %0s - result = %h", test_name, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected %h)", test_name, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_fp8_lane;
        input [1:0] lane;
        input [7:0] expected;
        input [255:0] test_name;
        reg [7:0] actual;
        begin
            test_count = test_count + 1;
            case (lane)
                2'd0: actual = result[7:0];
                2'd1: actual = result[15:8];
                2'd2: actual = result[23:16];
                2'd3: actual = result[31:24];
            endcase
            if (actual == expected) begin
                $display("PASS: %0s lane%0d - result = %h", test_name, lane, actual);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s lane%0d - result = %h (expected %h)", test_name, lane, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // FP8 E4M3 constants
    localparam FP8_ZERO = 8'h00;
    localparam FP8_ONE  = 8'h38;  // exp=7, man=0 -> 1.0 * 2^(7-7) = 1.0
    localparam FP8_TWO  = 8'h40;  // exp=8, man=0 -> 1.0 * 2^(8-7) = 2.0
    localparam FP8_HALF = 8'h30;  // exp=6, man=0 -> 1.0 * 2^(6-7) = 0.5
    localparam FP8_NEG1 = 8'hB8;  // -1.0

    initial begin
        $dumpfile("stage2_test.vcd");
        $dumpvars(0, tb_fma_fp8x4_stage2);

        rst_n = 0; valid_in = 0; mode = 0;
        A = 0; B = 0; C = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("=== Stage 2: FP32 + FP16 + FP8x4 FMA Testbench ===");

        // --- FP32 regression ---
        $display("--- FP32 Mode Regression ---");
        apply_input(2'b00, FP32_ONE, FP32_ONE, FP32_ZERO);
        wait_result();
        check_result(FP32_ONE, "FP32: 1*1+0");

        apply_input(2'b00, FP32_TWO, FP32_THREE, FP32_FOUR);
        wait_result();
        check_result(FP32_TEN, "FP32: 2*3+4");

        // --- FP16 regression ---
        $display("--- FP16 Mode Regression ---");
        apply_input(2'b01, FP16_ONE, FP16_ONE, FP16_ZERO);
        wait_result();
        check_result(FP16_ONE, "FP16: 1*1+0");

        apply_input(2'b01, FP16_TWO, FP16_THREE, FP16_FOUR);
        wait_result();
        check_result(FP16_TEN, "FP16: 2*3+4");

        // --- FP8x4 Tests ---
        $display("--- FP8x4 Mode Tests ---");

        // Test: all lanes 1.0 * 1.0 + 0.0 = 1.0
        apply_input(2'b10,
            pack_fp8(FP8_ONE, FP8_ONE, FP8_ONE, FP8_ONE),
            pack_fp8(FP8_ONE, FP8_ONE, FP8_ONE, FP8_ONE),
            pack_fp8(FP8_ZERO, FP8_ZERO, FP8_ZERO, FP8_ZERO));
        wait_result();
        check_fp8_lane(0, FP8_ONE,  "FP8: 1*1+0");
        check_fp8_lane(1, FP8_ONE,  "FP8: 1*1+0");
        check_fp8_lane(2, FP8_ONE,  "FP8: 1*1+0");
        check_fp8_lane(3, FP8_ONE,  "FP8: 1*1+0");

        // Test: all lanes 0.0 * 0.0 + 0.0 = 0.0
        apply_input(2'b10,
            pack_fp8(FP8_ZERO, FP8_ZERO, FP8_ZERO, FP8_ZERO),
            pack_fp8(FP8_ZERO, FP8_ZERO, FP8_ZERO, FP8_ZERO),
            pack_fp8(FP8_ZERO, FP8_ZERO, FP8_ZERO, FP8_ZERO));
        wait_result();
        check_fp8_lane(0, FP8_ZERO, "FP8: 0*0+0");
        check_fp8_lane(1, FP8_ZERO, "FP8: 0*0+0");
        check_fp8_lane(2, FP8_ZERO, "FP8: 0*0+0");
        check_fp8_lane(3, FP8_ZERO, "FP8: 0*0+0");

        // Test: lane0 = 1.0*1.0+1.0 = 2.0
        apply_input(2'b10,
            pack_fp8(FP8_ONE, FP8_ZERO, FP8_ZERO, FP8_ZERO),
            pack_fp8(FP8_ONE, FP8_ZERO, FP8_ZERO, FP8_ZERO),
            pack_fp8(FP8_ONE, FP8_ZERO, FP8_ZERO, FP8_ZERO));
        wait_result();
        check_fp8_lane(0, FP8_TWO, "FP8: 1*1+1=2");

        // Test: lane0 = 1.0 * (-1.0) + 0.0 = -1.0
        apply_input(2'b10,
            pack_fp8(FP8_ONE,  FP8_ZERO, FP8_ZERO, FP8_ZERO),
            pack_fp8(FP8_NEG1, FP8_ZERO, FP8_ZERO, FP8_ZERO),
            pack_fp8(FP8_ZERO, FP8_ZERO, FP8_ZERO, FP8_ZERO));
        wait_result();
        check_fp8_lane(0, FP8_NEG1, "FP8: 1*(-1)+0=-1");

        // Summary
        repeat(5) @(posedge clk);
        $display("=== Test Summary: %0d/%0d passed, %0d failed ===",
                 pass_count, test_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
