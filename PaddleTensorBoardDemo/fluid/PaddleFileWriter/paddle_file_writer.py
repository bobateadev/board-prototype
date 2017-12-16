import tensorflow as tf
import os
import time
import struct
from crc32c import crc32c


class PaddleFileWriter:

    def __init__(self, log_path = None):

        # tensorboard looks for tag "tfevents" in filename to load data
        filename = 'events.out.tfevents.{}'.format(int(time.time()))
        if log_path is None:
            path = filename
        else:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            path = os.path.join(log_path, filename)

        self.writer = open(path, 'wb')
        # every log file has to start with event of file version
        self.writeEvent(tf.Event(wall_time=time.time(), step=0, file_version='brain.Event:2'))

    #this function replicates scalar() function in tensorflow, simlpy logs a single value and plot in a graph
    def write(self, name, data, step = 0):
        # data will wrap in summary and write as a Event protobuf
        #'tag' will group the plot data in a single graph
        event = tf.Event(
            wall_time=time.time(),
            step=step,
            summary=tf.Summary(
                value=[tf.Summary.Value(
                    tag=name, simple_value=data)]))

        self.writeEvent(event)


    def write_graph(self, graph_def):
        # data will wrap in summary and write as a Event protobuf
        #'tag' will group the plot data in a single graph
        event = tf.Event(graph_def=graph_def.SerializeToString())
        self.writeEvent(event)


    def writeEvent(self, event):
        # serialize the protobuf as a string
        data = event.SerializeToString()
        w = self.writer
        # tensorboard uses a checksum algorithm(CRC) to verify data integrity

        #format defined in here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/record_writer.cc#L96

        # Format of a single record:
        # uint64    length
        # uint32    masked crc of length
        # byte      data[length]
        # uint32    masked crc of data

        # struck.pack will format string as binary data in a format
        header = struct.pack('Q', len(data)) #'Q' is the format of unsigned long long(uint64)
        w.write(header)
        w.write(struct.pack('I', masked_crc32c(header))) #'I' is unsigned int(uint32)
        w.write(data)
        w.write(struct.pack('I', masked_crc32c(data)))
        w.flush()


def masked_crc32c(data):
    # mast function defined in: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/hash/crc32c.h#L40
    kMaskDelta = 0xa282ead8
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + kMaskDelta)


def u32(x):
    return x & 0xffffffff
