module SCNN (
    input  wire        CLK,
    input  wire        Reset,
    input  wire        en,
    input  wire        CMD_en,
    //---------------------------
    input  wire [11:0] Data_in,
    input  wire        Valid_in,
    output wire        Ready_out,
    output wire [11:0] Data_out,
    output wire        Valid_out,
    input  wire        Ready_in
    //---------------------------
);


endmodule
