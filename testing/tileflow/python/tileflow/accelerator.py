import domino.accelerator as acc


__all__ = ["get_edge_small", "get_cloud_small"]


def get_edge_small():
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    L0 = acc.Buffer(name="L0", instance=8*8, buffer_class="SRAM", width=16, sizeKB=200,
                     meshX=8*8, word_bits=16, read_bandwidth=16, write_bandwidth=16, technology="16nm")
    PE = acc.Engine(name="PE")
    PE.add_local(L0, MAC)
    Core = acc.Engine(name="System")
    L1 = acc.Buffer(name="L1", buffer_class="DRAM", technology="16nm",
                    width=512, block_size=32, sizeKB=2000, word_bits=16)
    Core.add_level(PE)
    Core.add_local(L1)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_small():
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    L0 = acc.Buffer(name="L0", instance=32*32, buffer_class="SRAM", width=16, sizeKB=20000,
                     meshX=32*32, word_bits=16, read_bandwidth=400, write_bandwidth=400, technology="16nm")
    Part = acc.Engine(name="Part")
    Part.add_local(L0, MAC)
    subCore = acc.Engine(name="subCore")
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", read_bandwidth=20, write_bandwidth=20,
                    meshX=4*4, instance=4*4, technology="16nm",
                    width=512, block_size=32, sizeKB=40000, word_bits=16)
    subCore.add_level(Part)
    subCore.add_local(L1)
    Core = acc.Engine(name="System")
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    width=1024, block_size=32, sizeKB=40000000, word_bits=32)
    Core.add_level(subCore)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
