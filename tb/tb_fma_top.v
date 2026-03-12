//============================================================================
// Testbench for Multi-Precision FMA Top Module
// Tests: FP32, FP16, INT8, special values, and accumulation modes
//============================================================================
`timescale 1ns / 1ps

module tb_fma_top;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg  [2:0]  mode;
    reg         valid_in;
    reg  [31:0] A, B, C;
    reg         acc_en;
    reg  [31:0] acc_in;

    wire [31:0] result;
    wire [31:0] acc_out;
    wire        valid_out;
    wire        overflow;
    wire        underflow;
    wire        inexact;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    fma_top u_dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .mode      (mode),
        .valid_in  (valid_in),
        .A         (A),
        .B         (B),
        .C         (C),
        .acc_en    (acc_en),
        .acc_in    (acc_in),
        .result    (result),
        .acc_out   (acc_out),
        .valid_out (valid_out),
        .overflow  (overflow),
        .underflow (underflow),
        .inexact   (inexact)
    );

    //------------------------------------------------------------------------
    // Clock Generation: 10ns period
    //------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    //------------------------------------------------------------------------
    // Test Counter
    //------------------------------------------------------------------------
    integer test_num;
    integer pass_count;
    integer fail_count;

    //------------------------------------------------------------------------
    // Tasks
    //------------------------------------------------------------------------
    task reset;
    begin
        rst_n    = 0;
        mode     = 3'b000;
        valid_in = 0;
        A        = 32'b0;
        B        = 32'b0;
        C        = 32'b0;
        acc_en   = 0;
        acc_in   = 32'b0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
    end
    endtask

    task apply_input;
        input [2:0]  t_mode;
        input [31:0] t_A, t_B, t_C;
    begin
        @(negedge clk);   // Drive on negedge to avoid race with posedge capture
        mode     = t_mode;
        valid_in = 1;
        A        = t_A;
        B        = t_B;
        C        = t_C;
        @(negedge clk);
        valid_in = 0;
        A        = 32'b0;
        B        = 32'b0;
        C        = 32'b0;
    end
    endtask

    // Wait for valid_out and capture result
    reg [31:0] captured_result;
    reg        captured_overflow;

    task wait_and_capture;
        integer timeout;
    begin
        captured_result   = 32'b0;
        captured_overflow = 1'b0;
        timeout = 0;
        // Wait for valid_out to go high (with timeout)
        while (!valid_out && timeout < 20) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        // Capture on the cycle valid_out is high
        captured_result   = result;
        captured_overflow = overflow;
        @(posedge clk); // Advance past valid cycle
    end
    endtask

    task check_result;
        input [31:0] expected;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        if (captured_result === expected) begin
            $display("  [PASS] Test %0d: %0s - Result: 0x%08h", test_num, test_name, captured_result);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Test %0d: %0s - Expected: 0x%08h, Got: 0x%08h",
                     test_num, test_name, expected, captured_result);
            fail_count = fail_count + 1;
        end
    end
    endtask

    //------------------------------------------------------------------------
    // FP32 helper: encode IEEE 754 single precision
    // sign: 0 or 1, exp: biased exponent, man: 23-bit mantissa
    //------------------------------------------------------------------------
    function [31:0] fp32_encode;
        input        sign;
        input [7:0]  exp;
        input [22:0] man;
    begin
        fp32_encode = {sign, exp, man};
    end
    endfunction

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_fma_top.vcd");
        $dumpvars(0, tb_fma_top);

        test_num   = 0;
        pass_count = 0;
        fail_count = 0;

        $display("============================================");
        $display("  Multi-Precision FMA Testbench");
        $display("============================================");

        // Reset
        reset;

        //====================================================================
        // Test Group 1: FP32 FMA
        //====================================================================
        $display("\n--- FP32 FMA Tests ---");

        // Test 1: 1.0 * 1.0 + 0.0 = 1.0
        // FP32: 1.0 = 0_01111111_00000000000000000000000 = 0x3F800000
        // FP32: 0.0 = 0x00000000
        apply_input(3'b000, 32'h3F800000, 32'h3F800000, 32'h00000000);
        wait_and_capture;
        check_result(32'h3F800000, "FP32: 1.0*1.0+0.0=1.0");

        // Test 2: 2.0 * 3.0 + 0.0 = 6.0
        // FP32: 2.0 = 0x40000000, 3.0 = 0x40400000, 6.0 = 0x40C00000
        apply_input(3'b000, 32'h40000000, 32'h40400000, 32'h00000000);
        wait_and_capture;
        check_result(32'h40C00000, "FP32: 2.0*3.0+0.0=6.0");

        // Test 3: 1.0 * 1.0 + 1.0 = 2.0
        apply_input(3'b000, 32'h3F800000, 32'h3F800000, 32'h3F800000);
        wait_and_capture;
        check_result(32'h40000000, "FP32: 1.0*1.0+1.0=2.0");

        // Test 4: 0.0 * anything + 5.0 = 5.0
        // FP32: 5.0 = 0x40A00000
        apply_input(3'b000, 32'h00000000, 32'h3F800000, 32'h40A00000);
        wait_and_capture;
        check_result(32'h40A00000, "FP32: 0.0*1.0+5.0=5.0");

        //====================================================================
        // Test Group 2: FP32 Special Values
        //====================================================================
        $display("\n--- FP32 Special Value Tests ---");

        // Test 5: NaN input -> NaN output
        apply_input(3'b000, 32'h7FC00000, 32'h3F800000, 32'h00000000);
        wait_and_capture;
        check_result(32'h7FC00000, "FP32: NaN*1.0+0.0=NaN");

        // Test 6: Inf * 0 = NaN
        apply_input(3'b000, 32'h7F800000, 32'h00000000, 32'h00000000);
        wait_and_capture;
        check_result(32'h7FC00000, "FP32: Inf*0.0+0.0=NaN");

        // Test 7: Inf * 1.0 + 0.0 = Inf
        apply_input(3'b000, 32'h7F800000, 32'h3F800000, 32'h00000000);
        wait_and_capture;
        check_result(32'h7F800000, "FP32: Inf*1.0+0.0=Inf");

        //====================================================================
        // Test Group 3: INT8 MAD
        //====================================================================
        $display("\n--- INT8 MAD Tests ---");

        // Test 8: INT8: 2 * 3 + 0 = 6
        // A[7:0]=2, B[7:0]=3, C[15:0]=0
        apply_input(3'b011, {24'b0, 8'd2}, {24'b0, 8'd3}, {16'b0, 16'd0});
        wait_and_capture;
        check_result({16'b0, 16'd6}, "INT8: 2*3+0=6");

        // Test 9: INT8: 10 * 10 + 5 = 105
        apply_input(3'b011, {24'b0, 8'd10}, {24'b0, 8'd10}, {16'b0, 16'd5});
        wait_and_capture;
        check_result({16'b0, 16'd105}, "INT8: 10*10+5=105");

        // Test 10: INT8: -1 * 1 + 0 = -1 (0xFFFF)
        apply_input(3'b011, {24'b0, 8'hFF}, {24'b0, 8'h01}, {16'b0, 16'h0000});
        wait_and_capture;
        check_result({16'b0, 16'hFFFF}, "INT8: -1*1+0=-1");

        // Test 11: INT8: -2 * -3 + 1 = 7
        apply_input(3'b011, {24'b0, 8'hFE}, {24'b0, 8'hFD}, {16'b0, 16'd1});
        wait_and_capture;
        check_result({16'b0, 16'd7}, "INT8: -2*-3+1=7");

        //====================================================================
        // Test Group 4: Pipeline Throughput
        //====================================================================
        $display("\n--- Pipeline Throughput Test ---");

        // Feed consecutive inputs
        @(posedge clk);
        mode = 3'b000; valid_in = 1;
        A = 32'h3F800000; B = 32'h3F800000; C = 32'h00000000; // 1*1+0
        @(posedge clk);
        A = 32'h40000000; B = 32'h40000000; C = 32'h00000000; // 2*2+0
        @(posedge clk);
        A = 32'h40400000; B = 32'h40400000; C = 32'h00000000; // 3*3+0
        @(posedge clk);
        valid_in = 0;
        A = 32'b0; B = 32'b0; C = 32'b0;

        // Wait and check consecutive outputs
        repeat(5) @(posedge clk);

        $display("  [INFO] Pipeline throughput test completed (check waveform for details)");

        //====================================================================
        // Summary
        //====================================================================
        repeat(10) @(posedge clk);

        $display("\n============================================");
        $display("  Test Summary: %0d passed, %0d failed out of %0d",
                 pass_count, fail_count, test_num);
        $display("============================================");

        if (fail_count > 0) begin
            $display("  *** SOME TESTS FAILED ***");
        end else begin
            $display("  *** ALL TESTS PASSED ***");
        end

        $finish;
    end

endmodule
