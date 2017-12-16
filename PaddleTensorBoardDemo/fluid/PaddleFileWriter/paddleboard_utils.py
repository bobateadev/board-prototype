import collections

import tensorflow as tf


def convert_program_to_tf_graph_def(program):
    '''
    Crude implementation of PaddlePaddle Program to Tensorflow Graph.  Goes through each Paddle
    program block and creates a node for each variable and operation.  Linearly parse the inputs and
    outputs of each operation and create a connected Tensorflow Graph.

    This graph will later be logged to Tensorflow file, which will be rendered in Tensorboard
    :param program:  The PaddlePaddle Program object
    :return: Tensorflow Graph
    '''
    graph_def = tf.GraphDef()

    if len(program.blocks) > 0:
        op_name_counter = {}
        output_to_op_name = {}

        var_node_name_to_nodes = collections.OrderedDict()
        op_node_name_to_nodes = collections.OrderedDict()

        for block in program.blocks:
            # Process Program variables and create Tensorflow Nodes
            for var_name in block.vars:
                var_def = block.var(var_name)

                node_def = tf.NodeDef()
                node_def.name = var_def.name

                node_def.op = str(var_def.type)
                output_to_op_name[node_def.name] = node_def.name
                var_node_name_to_nodes[node_def.name] = node_def

            # Process Program operations and create Tensorflow Nodes
            for op in block.ops:
                attrs = {}

                if op.type in op_name_counter:
                    op_name_counter[op.type] += 1
                else:
                    op_name_counter[op.type] = 0

                node_name = '%s_%s' % (op.type, op_name_counter[op.type])

                for attr_name in op.desc.attr_names():
                    tensor_value = tf.AttrValue()
                    tensor_value.s = str(op.desc.attr(attr_name))
                    attrs[attr_name] = tensor_value

                inputs = []

                # Get operation inputs and outputs in order to create a connected Graph
                for input_name in op.input_names:
                    input_name = op.input(input_name)[0]
                    if input_name in output_to_op_name.keys():
                        input_node_name = output_to_op_name[input_name]
                        inputs.append(input_node_name)

                for output_name in op.output_names:
                    if op.output(output_name):
                        o_name = op.output(output_name)[0]
                        output_to_op_name[o_name] = node_name

                        if o_name in var_node_name_to_nodes.keys():
                            var_node = var_node_name_to_nodes[o_name]
                            var_node.input.extend([node_name])

                node_def = tf.NodeDef(attr=attrs, input=inputs)
                node_def.name = node_name

                node_def.op = op.type
                op_node_name_to_nodes[node_def.name] = node_def

        # Add Program variables and operations as Nodes in the Tensorflow graph
        graph_def.node.extend(var_node_name_to_nodes.values())
        graph_def.node.extend(op_node_name_to_nodes.values())

    return graph_def