import domino.accelerator as acc


__all__ = ["get_edge_small", "get_cloud_small",
           "get_edge_large", "get_cloud_large"]


def get_edge_small(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_edge_large(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=8000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_small(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_large(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=64, meshX=64)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=80000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=16, meshX=16)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
