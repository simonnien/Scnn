`timescale 1 ns / 1 ps
`celldefine
module SRAM_1024x13 (
    Q,
    CLK,
    CEN,
    WEN,
    A,
    D,
    EMA
);

    output [12:0] Q;
    input CLK;
    input CEN;
    input WEN;
    input [9:0] A;
    input [12:0] D;
    input [2:0] EMA;

endmodule
