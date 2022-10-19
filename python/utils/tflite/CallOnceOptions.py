# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class CallOnceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCallOnceOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CallOnceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def CallOnceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # CallOnceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CallOnceOptions
    def InitSubgraphIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def CallOnceOptionsStart(builder): builder.StartObject(1)
def CallOnceOptionsAddInitSubgraphIndex(builder, initSubgraphIndex): builder.PrependInt32Slot(0, initSubgraphIndex, 0)
def CallOnceOptionsEnd(builder): return builder.EndObject()
