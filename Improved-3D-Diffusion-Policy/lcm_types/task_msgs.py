import struct

class task_command_t(object):
    __slots__ = ["task_id"]
    def __init__(self):
        self.task_id = 0
    def encode(self):
        return struct.pack(">i", self.task_id)
    @classmethod
    def decode(cls, data):
        obj = cls()
        obj.task_id = struct.unpack(">i", data)[0]
        return obj

class task_result_t(object):
    __slots__ = ["success"]
    def __init__(self):
        self.success = 0
    def encode(self):
        return struct.pack(">i", self.success)
    @classmethod
    def decode(cls, data):
        obj = cls()
        obj.success = struct.unpack(">i", data)[0]
        return obj
