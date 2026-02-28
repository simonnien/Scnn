module CONV (
    input  wire        CLK,
    input  wire        en,
    //---------------------------
    input  wire [ 7:0] data,
    input  wire [ 9:0] addr,
    //---------------------------
    output wire [12:0] Weight,
    output wire [ 9:0] MP_addr,
    //---------------------------
    input  wire [ 1:0] k,
    input  wire [ 4:0] n,
    input  wire [ 1:0] S,
    //---------------------------
    input  wire [ 1:0] reg_en,
    output wire        next_Cin

);

endmodule
