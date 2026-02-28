module IOU (
    input  wire        CLK,
    input  wire        en,
    //---------------------------Master to IOU
    input  wire [11:0] Data_in,
    input  wire        Valid_in,
    output wire        Ready_out,
    output wire [11:0] Data_out,
    output wire        Valid_out,
    input  wire        Ready_in,
    //---------------------------IOU to (CU, CONV, LIF, PL)
    output wire [ 7:0] CMD,
    output wire [ 7:0] data,
    output wire [ 9:0] addr,
    //---------------------------Configuration Reg
    output reg  [ 4:0] data_n,
    output reg  [ 4:0] data_T,
    output reg  [ 9:0] data_Cin,
    output reg  [ 1:0] data_k,
    output reg  [ 1:0] data_S,
    output reg  [ 3:0] data_Beta,
    output reg  [11:0] data_Vth,
    output reg  [11:0] data_bias,
    //---------------------------PL to IOU
    input  wire        Spike,      //bits?
    input  wire        Spike_en,   //Protocol?
    //---------------------------IOU to CU
    input  wire [10:0] reg_en,
    output wire        buf_full
);


endmodule
